// SPDX-License-Identifier: Apache-2.0
// TensorRT V3 attention plugin. Sketch production/cache ownership stays outside
// the plugin; enqueue is allocation-free and uses TensorRT's stream exclusively.
#include "value_direction.h"
#include <NvInfer.h>
#include <array>
#include <cmath>
#include <cstring>
#include <new>

namespace vdtrt {
using namespace nvinfer1;
using fmha::value_direction::Params;
constexpr char NAME[]="ValueDirectionHopper";
constexpr char VERSION[]="2";
constexpr char SPACE[]="dllm";
struct Options {float scale=1.f, log_threshold=-INFINITY;int mode=1,precision=0,overlap=1,tma=0;};

class Plugin final:public IPluginV3,public IPluginV3OneCore,public IPluginV3OneBuild,public IPluginV3OneRuntime {
    Options options;
    std::array<PluginField,6> fields;
    PluginFieldCollection collection{};
public:
    explicit Plugin(Options o):options(o) {
        fields={PluginField{"scale",&options.scale,PluginFieldType::kFLOAT32,1},
                PluginField{"log_threshold",&options.log_threshold,PluginFieldType::kFLOAT32,1},
                PluginField{"mode",&options.mode,PluginFieldType::kINT32,1},
                PluginField{"precision",&options.precision,PluginFieldType::kINT32,1},
                PluginField{"overlap",&options.overlap,PluginFieldType::kINT32,1},
                PluginField{"tma",&options.tma,PluginFieldType::kINT32,1}};
        collection.nbFields=fields.size();collection.fields=fields.data();
    }
    IPluginV3* clone() noexcept override {return new(std::nothrow) Plugin(options);}
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch(type) {
            case PluginCapabilityType::kCORE:return static_cast<IPluginV3OneCore*>(this);
            case PluginCapabilityType::kBUILD:return static_cast<IPluginV3OneBuild*>(this);
            case PluginCapabilityType::kRUNTIME:return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    char const* getPluginName() const noexcept override{return NAME;}
    char const* getPluginVersion() const noexcept override{return VERSION;}
    char const* getPluginNamespace() const noexcept override{return SPACE;}
    int32_t getNbOutputs() const noexcept override{return 6;}
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs,int32_t ni,DynamicPluginTensorDesc const*,int32_t no) const noexcept override {
        if(ni!=6||no!=6||options.mode!=1||options.precision!=2)return 0;
        auto d=inputs[5].desc.dims;
        for(int i=0;i<4;++i)if(d.d[i]<0)d.d[i]=inputs[5].max.d[i];
        if(d.nbDims!=4||d.d[0]<1||d.d[1]<1||d.d[2]<1||d.d[3]<1)return 0;
        return size_t(d.d[0])*d.d[1]*d.d[2]*((d.d[3]+63)/64)*sizeof(uint64_t);
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const*,int32_t ni,DynamicPluginTensorDesc const*,int32_t no) noexcept override {
        return ni==6&&no==6?0:-1;
    }
    int32_t getOutputDataTypes(DataType* out,int32_t no,DataType const* in,int32_t ni) const noexcept override {
        if(ni!=6||no!=6)return -1;
        out[0]=in[0];out[1]=DataType::kBOOL;out[2]=DataType::kBOOL;
        out[3]=out[4]=out[5]=DataType::kFLOAT;return 0;
    }
    int32_t getOutputShapes(DimsExprs const* in,int32_t ni,DimsExprs const*,int32_t ns,
                           DimsExprs* out,int32_t no,IExprBuilder& b) noexcept override {
        if(ni!=6||no!=6||ns||in[0].nbDims!=4||in[1].nbDims!=4)return -1;
        out[0]=in[0];out[1]=in[0];
        out[1].d[2]=b.operation(DimensionOperation::kCEIL_DIV,*in[0].d[2],*b.constant(128));
        out[1].d[3]=b.operation(DimensionOperation::kCEIL_DIV,*in[1].d[2],*b.constant(64));
        out[2]=out[1];out[3]=in[0];out[3].nbDims=3;
        out[4]=in[0];out[4].d[3]=b.constant(32);out[5]=out[1];return 0;
    }
    bool supportsFormatCombination(int32_t pos,DynamicPluginTensorDesc const* io,int32_t ni,int32_t no) noexcept override {
        if(ni!=6||no!=6||pos<0||pos>=12)return false;
        auto const& d=io[pos].desc;
        DataType wanted=(pos<3||pos==6)?DataType::kBF16:
                        (pos==5||pos==7||pos==8)?DataType::kBOOL:DataType::kFLOAT;
        return d.format==TensorFormat::kLINEAR&&d.type==wanted;
    }
    int32_t onShapeChange(PluginTensorDesc const* in,int32_t ni,PluginTensorDesc const*,int32_t no) noexcept override {
        if(ni!=6||no!=6)return -1;
        for(int x:{0,1,2,3,5})if(in[x].dims.nbDims!=4)return -1;
        if(in[4].dims.nbDims!=2)return -1;
        auto q=in[0].dims,k=in[1].dims;
        if(q.d[0]<1||q.d[1]<1||q.d[2]<1||k.d[1]<1||k.d[2]<1||q.d[1]%k.d[1]||
           k.d[0]!=q.d[0]||k.d[3]!=q.d[3]||(q.d[3]!=256&&q.d[3]!=512))return -1;
        for(int x=0;x<4;++x)if(in[2].dims.d[x]!=k.d[x]||in[3].dims.d[x]!=(x==3?32:k.d[x]))return -1;
        if(in[4].dims.d[0]!=q.d[0]||in[4].dims.d[1]!=k.d[1])return -1;
        auto m=in[5].dims;
        if(m.d[0]!=q.d[0]||(m.d[1]!=1&&m.d[1]!=q.d[1])||m.d[2]!=q.d[2]||m.d[3]!=k.d[2])return -1;
        return 0;
    }
    int32_t enqueue(PluginTensorDesc const* in,PluginTensorDesc const*,void const* const* inputs,
                    void* const* outputs,void* workspace,cudaStream_t stream) noexcept override {
        Params p{};
        p.q=inputs[0];p.k=inputs[1];p.v=inputs[2];p.z=static_cast<float const*>(inputs[3]);
        p.reference=static_cast<float const*>(inputs[4]);p.mask=inputs[5];p.output=outputs[0];
        p.skipped=static_cast<uint8_t*>(outputs[1]);p.eligible=static_cast<uint8_t*>(outputs[2]);
        p.log_normalizer=static_cast<float*>(outputs[3]);p.projected_state=static_cast<float*>(outputs[4]);p.risks=static_cast<float*>(outputs[5]);
        p.batch=in[0].dims.d[0];p.heads=in[0].dims.d[1];p.queries=in[0].dims.d[2];p.width=in[0].dims.d[3];
        p.kv_heads=in[1].dims.d[1];p.keys=in[1].dims.d[2];p.mask_kind=1;p.mask_heads=in[5].dims.d[1];
        p.mode=options.mode;p.trace=1;p.scale=options.scale;p.log_threshold=options.log_threshold;p.projected_precision=options.precision;
        if(options.mode==1&&options.precision==2) {
            if(!workspace)return -1;
            auto status=value_direction_pack_mask(static_cast<uint8_t const*>(p.mask),static_cast<uint64_t*>(workspace),p.batch,p.mask_heads,p.queries,p.keys,stream);
            if(status!=cudaSuccess)return -1;
            p.mask=workspace;p.mask_kind=3;
        }
        int overlap=options.overlap==2&&p.width!=512?1:options.overlap;
        return (options.tma?value_direction_sm90_tma(&p,stream,overlap):value_direction_sm90(&p,stream,overlap))==cudaSuccess?0:-1;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override{return clone();}
    PluginFieldCollection const* getFieldsToSerialize() noexcept override{return &collection;}
};

class Creator final:public IPluginCreatorV3One {
    std::array<PluginField,6> fields={PluginField{"scale",nullptr,PluginFieldType::kFLOAT32,1},
        PluginField{"log_threshold",nullptr,PluginFieldType::kFLOAT32,1},PluginField{"mode",nullptr,PluginFieldType::kINT32,1},
        PluginField{"precision",nullptr,PluginFieldType::kINT32,1},PluginField{"overlap",nullptr,PluginFieldType::kINT32,1},PluginField{"tma",nullptr,PluginFieldType::kINT32,1}};
    PluginFieldCollection collection{int32_t(fields.size()),fields.data()};
public:
    char const* getPluginName() const noexcept override{return NAME;}
    char const* getPluginVersion() const noexcept override{return VERSION;}
    char const* getPluginNamespace() const noexcept override{return SPACE;}
    PluginFieldCollection const* getFieldNames() noexcept override{return &collection;}
    IPluginV3* createPlugin(char const*,PluginFieldCollection const* fc,TensorRTPhase) noexcept override {
        if(!fc||value_direction_abi_version()!=fmha::value_direction::ABI_VERSION||value_direction_params_size()!=sizeof(Params))return nullptr;
        Options o;
        for(int i=0;i<fc->nbFields;++i){auto const& f=fc->fields[i];
            if(!f.name||!f.data||f.length!=1)return nullptr;
            if(!std::strcmp(f.name,"scale")&&f.type==PluginFieldType::kFLOAT32)std::memcpy(&o.scale,f.data,4);
            else if(!std::strcmp(f.name,"log_threshold")&&f.type==PluginFieldType::kFLOAT32)std::memcpy(&o.log_threshold,f.data,4);
            else if(!std::strcmp(f.name,"mode")&&f.type==PluginFieldType::kINT32)std::memcpy(&o.mode,f.data,4);
            else if(!std::strcmp(f.name,"precision")&&f.type==PluginFieldType::kINT32)std::memcpy(&o.precision,f.data,4);
            else if(!std::strcmp(f.name,"overlap")&&f.type==PluginFieldType::kINT32)std::memcpy(&o.overlap,f.data,4);
            else if(!std::strcmp(f.name,"tma")&&f.type==PluginFieldType::kINT32)std::memcpy(&o.tma,f.data,4);
            else return nullptr;
        }
        if(!std::isfinite(o.scale)||std::isnan(o.log_threshold)||o.mode<0||o.mode>2||o.precision<0||o.precision>2||o.overlap<0||o.overlap>2||o.tma<0||o.tma>1)return nullptr;
        if(o.tma&&(o.mode!=1||o.precision!=2||!o.overlap))return nullptr;
        return new(std::nothrow) Plugin(o);
    }
};
extern "C" bool init_value_direction_plugin() {
    static Creator creator;
    static bool registered=getPluginRegistry()->registerCreator(creator,SPACE);
    return registered;
}
} // namespace vdtrt

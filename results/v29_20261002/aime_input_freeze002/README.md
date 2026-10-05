# AIME input freeze attempts

Source `1931db5a6540409946641ebb0b04c06507d48402`: attempt001 preserved the transmitted rows but rejected their byte pin because Windows CRLF input was reserialized with Linux LF. It stopped before binding, launched no GPU, and its failed directory remains intact.

Attempt002 transmitted the exact original minimal manifest bytes, checked their original private SHA and parsed equality, then froze and verified the binding successfully (both exit0). All30 rows retain only id, tokenized input,8192 budget and thinking metadata. Gold contents remain on the existing CPU scorer machine; only scoring provenance metadata are bound. No config/provenance files were retransferred in this input stage. Raw text, answers, private hashes and paths are omitted here.

Both attempts used0 GPU seconds. Binding readiness is a CPU freeze result; four-arm GPU qualification and scored proof are separate gates.

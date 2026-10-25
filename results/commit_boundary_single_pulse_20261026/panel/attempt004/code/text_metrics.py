"""Frozen descriptive text rubric; no content or answer strings are published."""
import ast
import re
from collections import Counter


def metrics(text):
    words=re.findall(r'\S+',text)
    grams=[tuple(words[i:i+8]) for i in range(max(0,len(words)-7))]
    counts=Counter(grams)
    lines=[re.sub(r'\s+',' ',x).strip() for x in text.splitlines() if x.strip()]
    markers=[m.start() for m in re.finditer(re.escape('<channel|>'),text)]
    final_start=markers[-1]+len('<channel|>') if markers else None
    # Observable revision vocabulary is a text property, not a reasoning label.
    cues=re.findall(r'\b(?:wait|actually|reconsider|start over|correction|instead|let me check)\b',text,re.I)
    repeated_defs=0;parseable_fences=0
    for body in re.findall(r'```(?:python\w*|py)?\s*\n(.*?)```',text,re.S|re.I):
        try:tree=ast.parse(body)
        except (SyntaxError,ValueError):continue
        names=[x.name for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))]
        repeated_defs+=len(names)-len(set(names));parseable_fences+=1
    # Conservative AIME endpoint: a closed boxed expression in the final channel.
    # This is retrospective character location, not native stopping or correctness.
    boxed=None
    if final_start is not None:
        match=re.search(r'\\boxed\{[^{}]*\}',text[final_start:])
        if match:boxed=final_start+match.end()
    return dict(characters=len(text),whitespace_words=len(words),nonempty_lines=len(lines),
        duplicate_line_fraction=(len(lines)-len(set(lines)))/max(1,len(lines)),
        repeated_8gram_fraction=sum(n-1 for n in counts.values())/max(1,len(grams)),
        revision_vocabulary_count=len(cues),final_channel_markers=len(markers),
        final_channel_character=final_start,first_closed_box_in_final_character=boxed,
        parseable_python_fences=parseable_fences,repeated_function_definitions=repeated_defs,
        end_markers=sum(text.count(x) for x in ['<turn|>','<|endoftext|>','<eos>']))

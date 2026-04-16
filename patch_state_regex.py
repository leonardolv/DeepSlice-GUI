import re

file_path = "DeepSlice/gui/state.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add is_dirty flag
content = re.sub(
    r"class DeepSliceAppState:\n",
    "class DeepSliceAppState:\n    is_dirty: bool = False\n",
    content,
    count=1
)

# 2. Add self.is_dirty = True to mutating methods
pattern_true = re.compile(
    r"(^[ \t]+def\s+(set_images|add_images|clear_images|undo|redo|run_prediction|set_bad_sections|apply_manual_order|propagate_angles|adjust_angles|enforce_index_order|enforce_index_spacing)\b[^:]*:\s*\n)",
    re.MULTILINE
)
content = pattern_true.sub(r"\1        self.is_dirty = True\n", content)

# 3. Add self.is_dirty = False to loading methods
pattern_false = re.compile(
    r"(^[ \t]+def\s+(load_quint|load_session_dict)\b[^:]*:\s*\n)",
    re.MULTILINE
)
content = pattern_false.sub(r"\1        self.is_dirty = False\n", content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied.")

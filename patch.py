import os

file_path = "DeepSlice/gui/state.py"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("class DeepSliceAppState:"):
        new_lines.append(line)
        new_lines.append("    is_dirty: bool = False\n")
        continue

    new_lines.append(line)

    if line.startswith("    def set_images("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def add_images("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def clear_images("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def undo("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def redo("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def run_prediction("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def load_quint("):
        new_lines.append("        self.is_dirty = False\n")
    elif line.startswith("    def load_session_dict("):
        new_lines.append("        self.is_dirty = False\n")
    elif line.startswith("    def set_bad_sections("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def apply_manual_order("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def propagate_angles("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def adjust_angles("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def enforce_index_order("):
        new_lines.append("        self.is_dirty = True\n")
    elif line.startswith("    def enforce_index_spacing("):
        new_lines.append("        self.is_dirty = True\n")

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

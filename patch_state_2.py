import os

file_path = "DeepSlice/gui/state.py"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
inside_method = None

methods_true = {"set_images", "add_images", "run_prediction", "set_bad_sections", "apply_manual_order", "adjust_angles", "enforce_index_spacing"}
methods_false = {"load_quint", "load_session_dict"}

for line in lines:
    new_lines.append(line)
    
    # Check if this line is the END of a method signature (ends with :)
    if inside_method and line.strip().endswith(":"):
        if inside_method in methods_true:
            new_lines.append("        self.is_dirty = True\n")
        elif inside_method in methods_false:
            new_lines.append("        self.is_dirty = False\n")
        inside_method = None
        continue

    # Check if this line starts a method signature we care about
    if line.strip().startswith("def "):
        method_name = line.strip()[4:].split("(")[0].strip()
        if method_name in methods_true or method_name in methods_false:
            if line.strip().endswith(":"):
                # One-liner signature
                if method_name in methods_true:
                    new_lines.append("        self.is_dirty = True\n")
                elif method_name in methods_false:
                    new_lines.append("        self.is_dirty = False\n")
            else:
                # Multi-line signature
                inside_method = method_name

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Patch 2 applied.")

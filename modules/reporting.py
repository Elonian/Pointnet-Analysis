from pathlib import Path
from typing import Any

from modules.runtime import write_json


def write_markdown_report(path: str | Path, title: str, sections: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for name, value in sections.items():
        lines.extend([f"## {name}", ""])
        if isinstance(value, dict):
            for key, item in value.items():
                lines.append(f"- `{key}`: {item}")
        elif isinstance(value, list):
            if not value:
                lines.append("- No entries.")
            else:
                for item in value:
                    lines.append(f"- {item}")
        else:
            lines.append(str(value))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report_bundle(output_dir: str | Path, name: str, payload: dict[str, Any], markdown_sections: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    write_json(output_dir / f"{name}.json", payload)
    write_markdown_report(output_dir / f"{name}.md", name.replace("_", " ").title(), markdown_sections)


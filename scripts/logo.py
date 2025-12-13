from pathlib import Path

def generate_map_and_play_logo(output_file: str, width: int = 400, height: int = 400) -> None:
    """
    Generates a static SVG logo with a map and play button.

    Args:
        output_file: The name of the output file.
        width: The width of the SVG image.
        height: The height of the SVG image.
    """
    center_x = width / 2
    center_y = height / 2

    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#4a90e2;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#003c8b;stop-opacity:1" />
        </linearGradient>
    </defs>
    <rect x="0" y="0" width="{width}" height="{height}" rx="20" ry="20" fill="url(#grad1)" />
    
    <!-- Simplified world map -->
    <path d="M{width*0.1} {height*0.5} Q{width*0.3} {height*0.2}, {width*0.5} {height*0.5} T{width*0.9} {height*0.5}" stroke="white" stroke-width="8" fill="none" stroke-linecap="round" />
    <path d="M{width*0.2} {height*0.3} Q{width*0.5} {height*0.4}, {width*0.8} {height*0.3}" stroke="white" stroke-width="8" fill="none" stroke-linecap="round" />
    <path d="M{width*0.2} {height*0.7} Q{width*0.5} {height*0.6}, {width*0.8} {height*0.7}" stroke="white" stroke-width="8" fill="none" stroke-linecap="round" />

    <!-- Play button -->
    <polygon points="{center_x-50},{center_y-50} {center_x-50},{center_y+50} {center_x+50},{center_y}" style="fill:rgba(255, 255, 255, 0.9);stroke:white;stroke-width:5" />
</svg>
"""

    with open(output_file, "w") as f:
        f.write(svg_content)

    print(f"Static SVG logo created: {output_file}")

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "_static" / "logo.svg"
    generate_map_and_play_logo(output_path)

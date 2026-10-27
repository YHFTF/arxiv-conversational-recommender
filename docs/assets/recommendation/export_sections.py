import json, re, html
from pathlib import Path
import xml.etree.ElementTree as ET
root=Path(__file__).resolve().parent/'svg'
sections=json.loads((root/'layout.json').read_text(encoding='utf-8'))
def esc(v): return html.escape(str(v),quote=True)
def color(v):
    if v in ('transparent','rgba(0, 0, 0, 0)'): return 'none'
    if v.startswith('rgb('):
        nums=[int(x) for x in re.findall(r'\d+',v)]
        return '#%02x%02x%02x'%tuple(nums[:3])
    return v
for section in sections:
    defs=[]; body=[]
    w,h=section['width'],section['height']
    body.append(f'<rect width="{w}" height="{h}" rx="14" fill="#090c12"/>')
    for item in section['items']:
        x,y=item['x'],item['y']
        if item['type']=='text':
            body.append(f'<text x="{x:.3f}" y="{y:.3f}" fill="{color(item["fill"])}" font-size="{item["size"]}" font-weight="{item["weight"]}" xml:space="preserve">{esc(item["text"])}</text>')
        else:
            fill=color(item['fill']); stroke=color(item['border']) if item['bw'] else 'none'
            gradient=item['gradient']
            if gradient.startswith('linear-gradient'):
                colors=re.findall(r'rgba?\([^)]+\)|#[\da-fA-F]+',gradient)
                if len(colors)>=2:
                    gid=f'g{len(defs)}'
                    defs.append(f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1"><stop stop-color="{color(colors[0])}"/><stop offset="1" stop-color="{color(colors[-1])}"/></linearGradient>')
                    fill=f'url(#{gid})'
            if fill!='none' or stroke!='none':
                radius=min(item['radius'] if item['radius'] is not None else item['h']/2,item['w']/2,item['h']/2)
                body.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{item["w"]:.3f}" height="{item["h"]:.3f}" rx="{radius:.3f}" fill="{fill}" stroke="{stroke}" stroke-width="{item["bw"]}"/>')
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Malgun Gothic, sans-serif"><title>{esc(section["name"])}</title><defs>{"".join(defs)}</defs>{"".join(body)}</svg>'
    target=root/(section['name']+'.svg')
    target.write_text(svg,encoding='utf-8')
    ET.parse(target)
    assert '<image' not in svg and '<foreignObject' not in svg
    print(f'{target}: {w:.0f} x {h:.0f}, {target.stat().st_size:,} bytes; vector-only SVG validated')


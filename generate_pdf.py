import subprocess
import markdown

with open("SPRAWOZDANIE.md", "r", encoding="utf-8") as f:
    md_content = f.read()

html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

css = """
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    line-height: 1.5;
    color: #222;
    margin: 40px auto;
    max-width: 900px;
    padding: 0 20px;
}
h1 { color: #1a365d; border-bottom: 2px solid #2b6cb0; padding-bottom: 8px; margin-top: 30px; font-size: 24px; }
h2 { color: #2c5282; font-size: 20px; margin-top: 25px; }
h3 { color: #2b6cb0; font-size: 17px; margin-top: 20px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
h4 { color: #4a5568; font-size: 15px; margin-top: 15px; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 13px;
}
th, td {
    border: 1px solid #cbd5e0;
    padding: 6px 10px;
    text-align: left;
}
th {
    background-color: #ebf8ff;
    color: #2b6cb0;
    font-weight: 600;
}
tr:nth-child(even) {
    background-color: #f7fafc;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 15px auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
}
code {
    background-color: #edf2f7;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: Consolas, monospace;
    font-size: 12px;
}
ul, ol {
    padding-left: 20px;
}
li {
    margin-bottom: 4px;
}
hr {
    border: 0;
    border-top: 1px solid #e2e8f0;
    margin: 25px 0;
}
@media print {
    body {
        margin: 10mm;
        padding: 0;
        font-size: 11pt;
    }
    table {
        page-break-inside: avoid;
        font-size: 10pt;
    }
    img {
        page-break-inside: avoid;
        max-width: 90%;
    }
    h1, h2, h3, h4 {
        page-break-after: avoid;
    }
}
"""

html_full = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Sprawozdanie - Laboratorium 2</title>
<style>
{css}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open("SPRAWOZDANIE.html", "w", encoding="utf-8") as f:
    f.write(html_full)

print("Wygenerowano SPRAWOZDANIE.html.")

# Konwersja do PDF przez headless Edge
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
import os
html_abs = os.path.abspath("SPRAWOZDANIE.html")
pdf_abs = os.path.abspath("SPRAWOZDANIE.pdf")

cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    f"--print-to-pdf={pdf_abs}",
    html_abs
]

res = subprocess.run(cmd, capture_output=True, text=True)
if os.path.exists(pdf_abs) and os.path.getsize(pdf_abs) > 1000:
    print(f"Pomyślnie wygenerowano SPRAWOZDANIE.pdf (rozmiar: {os.path.getsize(pdf_abs)} bajtów).")
else:
    print("Nie udało się utworzyć PDF przez Edge:", res.stderr)

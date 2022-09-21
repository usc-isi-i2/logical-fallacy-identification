"""
Span identification post-processing.
Merges predicted spans that are separated by short gaps.
"""

SPANS_FILE = 'spans.txt'
MIN_GAP_LENGTH = 25

prev_article = -1
entries = []
with open(SPANS_FILE, 'r', encoding='utf8') as f:
    for line in f:
        cells = line.split('\t')
        article = cells[0]
        start = int(cells[1])
        end = int(cells[2])
        if prev_article == article and start - entries[-1][2] < MIN_GAP_LENGTH:
            entries[-1] = (article, entries[-1][1], end)
        else:
            entries.append((article, start, end))
        prev_article = article

with open('post_' + SPANS_FILE, 'w', encoding='utf8') as f:
    for article, start, end in entries:
        f.write(article + '\t' + str(start) + '\t' + str(end) + '\n')

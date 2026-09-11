# Long-context fixture

`tale-of-two-cities.txt.gz` contains the complete original Project Gutenberg UTF-8 text of Charles Dickens's *A Tale of Two Cities* (ebook 98), including the Gutenberg header and license text. It is the exact raw input used in the recorded 2026-09-11 experiment.

Source: https://www.gutenberg.org/ebooks/98.txt.utf-8

Run `python3 scripts/prepare_book.py` from the repository root to extract it and produce the cleaned benchmark input. Raw/clean SHA-256 values are recorded in `measurements/2026-09-11/book-source.json`. Compression uses a zero gzip timestamp for reproducible bytes. Preserve the complete raw file and its license/header when redistributing this fixture.

The novel itself is public domain in the United States and many other jurisdictions. The Project Gutenberg license included in the original text sets out terms for use of its name and distributions. The cookbook's Apache license does not replace those terms.

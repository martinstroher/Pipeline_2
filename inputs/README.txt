# Pipeline inputs

Place source PDF or Markdown documents in this directory. The standard pipeline
converts PDFs to Markdown and builds its retrieval index from the Markdown
files.

The tracked `mock_*.pdf` files are short, wholly synthetic smoke-test inputs.
They contain no article text and make no scientific claims. Remove them before
adding a real corpus so their terms do not enter a production run.

User-supplied corpus files are ignored by Git. Do not commit publisher PDFs or
other documents unless you have explicit redistribution rights.

For a one-document or two-document smoke test, set
`MINIMUM_FREQUENCY_FILTER=1`; the production default of 5 distinct documents
would intentionally filter out every term from such a small example.

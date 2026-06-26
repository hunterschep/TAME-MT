# Security And Privacy

TAME-MT runs locally and does not send data to remote services.

## Reporting Security Issues

Please report vulnerabilities privately through GitHub security advisories for
the repository when available. If that is not available, contact the maintainer
directly before opening a public issue.

## Data Handling

Training corpora can contain sensitive, licensed, community-owned, or otherwise
restricted text. TAME-MT's default behavior avoids printing nearest-neighbor
training examples and avoids including raw segment text in JSONL diagnostics.

The following flags can write raw text and should be used only when appropriate:

```bash
--include-source-text
--include-reference-text
--include-hyp-text
--include-neighbor-text
```

`--include-neighbor-text` can write raw training text.

# TAME-MT Toy Example

This tiny corpus is designed to exercise all default source-exposure bins:
an exact training pair, a near training example, a medium example, and a far
example.

Run:

```bash
tame-mt score \
  --train-src examples/toy/train.src \
  --train-tgt examples/toy/train.tgt \
  --test-src examples/toy/test.src \
  --ref examples/toy/test.ref \
  --hyp examples/toy/hyp.out
```

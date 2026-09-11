# Earthrise vision fixtures

The original photograph is **Earthrise**, taken by Bill Anders during NASA's Apollo 8 mission in 1968.

- Source: https://upload.wikimedia.org/wikipedia/commons/a/a8/NASA-Apollo8-Dec24-Earthrise.jpg
- Description and rights: https://commons.wikimedia.org/wiki/File:NASA-Apollo8-Dec24-Earthrise.jpg
- Credit: NASA / Bill Anders. US federal government work, public domain.

`earthrise.jpg` preserves the downloaded bytes. SHA-256 values and source details are in `source.json`. The model receives image bytes through data URLs, not the filename, source URL or photograph title.

Controls generated for this test:

- `earthrise-1024.png`: original resized to 1024 × 1024.
- `earthrise-flipped.png`: vertically flipped; rocky terrain is now at the top.
- `lunar-crop.png`: bottom 24% of the resized image; Earth is absent.
- `earthrise-marked.png`: resized original with the exact text `RAVEN 6247` at top left and a red square at top right.

These changes distinguish visible-image grounding from recognition of a famous photograph. The marked image is a synthetic test derivative, not an unmodified NASA photograph. Concurrent original/crop requests check image identity isolation; repeated original requests exercise cache reuse. A two-image request checks ordering and comparison.

## Video controls

`sequence.mp4` and `sequence-reversed.mp4` are six-second, 12-fps generated derivatives of this photograph. The forward sequence shows ALPHA/red/left, BRAVO/green/center, and CHARLIE/blue/right in two-second stages. The other clip reverses every frame. They test temporal order, not recognition of a known movie. Video generation details and SHA-256 values are in `video-source.json`.

The real-video test downloads OpenCV's public vtest.avi sample separately using `scripts/prepare_video.py`; it is not bundled here. See that sample's upstream terms when redistributing it.

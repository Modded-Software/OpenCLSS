# Settings and frame times

The daemon re-reads its settings file whenever it changes, so every setting below moves
between frames while the game runs. `src/layer/nr-panel`, `src/layer/nr-ctl` and
`src/layer/nr-toggle` all write the same file. The blocks below are generated from
`src/layer/nr_knobs.py` by `src/tools/knob_doc.py`.

## Settings

<!-- knobs:begin -->

### `render_scale` - the fraction of each side the network runs on

`0.05` to `1`, step `0.05`, default `1`

The only knob that changes the frame rate. The network draws its detail on a frame this much smaller; the detail is then scaled up and laid over the game's full-resolution frame, so the game's own pixels are never resampled. Lower is faster and draws coarser detail, at about the same strength. The network's frame is never smaller than 320 pixels a side (`min_extent`), so on a small window a low scale is raised, at no cost, to fill the cheapest frame it lands on: 640x360 runs as 0.5 for anything up to 0.5, 800x450 as 0.4 for anything up to 0.4, and a lower scale is never the slower one. For play, 0.35-0.6 is the useful range. For screenshots, 1.0 hands the network the game's raw pixels, jagged edges and all, and it turns a little of them into pixel-level grain; 0.9 smooths them first and draws the same strength of effect. Cost on an Arc 140V: about 130 ms per megapixel of network frame, plus 10-20 ms that grows with the game's own resolution.

### `min_extent` - the smallest side the network's frame is padded to

`128` to `320`, step `64`, default `320`

The network's frame is padded, by mirroring the picture, to at least this many pixels on a side. 320 is what NVIDIA's own driver does; the network itself runs down to 128. At small live sizes most of a 320 frame is padding, so a lower floor is much faster - on an Arc 140V, 512x288 at scale 0.35 takes 25 ms a frame at 320 and 14 at 128 - and draws a somewhat different picture, since the network no longer sees a mirrored copy of the scene around it. Neither is wrong; compare them in a game. It changes nothing once the scaled frame is larger than this anyway.

### `profile` - which way to trade skin texture against highlights, and its colour grade

`standard` / `natural` / `cinematic` / `neutral`

The style the network is asked for, and the colour grade that comes with it. The profiles are a trade, not a quality ladder: what one adds to skin and surface texture it takes from highlights. `standard` is the default and adds the most texture; `natural` and `cinematic` keep more of the highlights, and on very bright scenes `cinematic` can smooth fine detail rather than add it. After the network, as NVIDIA grades them, `natural` comes out a little darker, flatter and less saturated and `cinematic` less saturated. `neutral` all but switches the effect off. The profile is an input to the network, so it takes effect on the next frame the network draws; the knobs below act after it.

### `intensity` - how far to go towards the model's picture, or past it

`0` to `2`, step `0.05`, default `1`

Blends the model's picture with the game's own. 1 is the model's picture, 0 is the game's frame untouched, and values between are part of the way. Above 1 the blend goes past the model and exaggerates what it changed, which can look overdone. Free to change: nothing is re-run.

### `detail_strength` - how strongly to apply the fine detail the pass adds

`0` to `2`, step `0.05`, default `1`

The change the pass makes is split into fine detail - pores, hair, grain - and broad tone. This scales the fine part: 0 keeps only the tonal change, above 1 sharpens further, and past about 2 it over-sharpens. Away from 1 it costs one blur over the whole frame.

### `colour_strength` - how strongly to apply the pass's change of tone and colour

`0` to `2`, step `0.05`, default `1`

The broad half of the same split: how much of the pass's change in tone and colour is applied. It does not add colour back. The pass tends to calm bright, saturated areas, and this scales that change - so above 1 colours can look more washed out, not richer, and 0 keeps the game's own tone with the detail on top.

### `temporal` - how much of the previous frame to carry over

`0` to `1`, step `0.05`, default `1`

Each frame, the previous result is fed back to the network, which decides per pixel how much of it to keep; that is what keeps the picture from shimmering. This scales how much it keeps. 0 draws every frame on its own and forgets the stored frame, so turning it back on cannot bring back an old one. Costs a few per cent of the frame time.

### `hold` - how firmly to keep areas the game did not change

`0` to `1`, step `0.05`, default `1`

Where the game hands back exactly the same pixel as last frame, the previous result is still right for it, so it is kept at least this firmly. This is what stops still areas - backgrounds, a waiting character - from shimmering while something else moves. It cannot leave a trail: the first frame in which the game changes a pixel lets go of it.

### `release` - change, in levels of 255, that drops the previous frame where something moved

`0` to `64`, step `1`, default `24`

The other side of `hold`. Where the game's own pixel changed by this many levels of 255 or more, something moved there, and the previous result is dropped for that pixel; smaller changes keep a share that falls with the change. It is what prevents trails behind moving objects, since the layer has no motion vectors to follow them with. 16-24 is the useful range: lower drops more and the picture starts to shimmer over moving things, 0 turns it off and trails come back.

### `cut_limit` - how big a change between frames counts as a new scene

`0` to `1`, step `0.01`, default `0.15`

The average change between two frames above which the scene is taken to have cut - a camera cut, a menu, a replay - and the previous frame is dropped entirely instead of pixel by pixel. Lower cuts more readily; 1 never cuts.

<!-- knobs:end -->

## Frame times

<!-- rates:begin -->

Measured through the socket on 2026-09-27 by `uv run python src/bench/live_rates.py` - the whole round trip a game waits for, median of nine frames, not graph time alone:

| swapchain | render scale | ms | fps |
| --- | ---: | ---: | ---: |
| 512x288 | 0.35 | 26 | 38.6 |
| 512x288 | 0.50 | 26 | 38.5 |
| 640x360 | 0.35 | 26 | 38.5 |
| 640x360 | 0.50 | 27 | 37.0 |
| 854x480 | 0.50 | 34 | 29.4 |
| 1024x768 | 0.55 | 56 | 17.7 |
| 1920x1080 | 0.55 | 112 | 8.9 |

Medians of six runs with swap empty; a row's runs spread up to 15 %. On 2026-09-23, with 5.5 GiB in zram and the kernel's memory-pressure figures rising, 1920x1080 ran anywhere from 322 to 463 ms: if that row is much slower for you, look at swap before anything else.

That is the daemon's own cost with nothing else on the GPU. A game adds its own frame to it: **Tekken 7** ran at **30 fps at 800x450** with the render scale at 0.35, and 27 at 0.6, in a live session on 2026-09-27 - against 10.5 fps at 640x360 on 2026-09-16 (`notes/phase59`).

<!-- rates:end -->

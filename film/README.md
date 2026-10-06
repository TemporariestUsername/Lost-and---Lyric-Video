# lost and: the film

The finished lyric video (8:14, 30 fps, stereo AAC), rendered from this branch after the review pass (italic rule, the swap, But, the bend, one surgical language).

- [`lost_and_1080p.mp4`](lost_and_1080p.mp4): 1920x1080, about 600 MB. The one to keep.
- [`lost_and_720p.mp4`](lost_and_720p.mp4): 1280x720, about 180 MB. For phones or sharing.

On GitHub, open a file and press **Download raw file**. Both are stored with Git LFS, so a plain clone needs `git lfs pull` to fetch them.

Re-make them from the master (`render/lost_and_full.mp4`, which isn't committed):

    ffmpeg -i render/lost_and_full.mp4 -c:v libx264 -preset slow -b:v 9500k -pass 1 -an -f null /dev/null
    ffmpeg -i render/lost_and_full.mp4 -c:v libx264 -preset slow -b:v 9500k -maxrate 14000k -bufsize 20000k -pass 2 \
        -pix_fmt yuv420p -c:a aac -b:a 256k -movflags +faststart film/lost_and_1080p.mp4
    python3 video/web.py render/lost_and_full.mp4
    ffmpeg -allowed_extensions ALL -i storyboard/renders/film/index.m3u8 -c copy -movflags +faststart film/lost_and_720p.mp4

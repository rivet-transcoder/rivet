set -e
FF=/usr/local/bin/ffmpeg
O=/tmp/corpus
mkdir -p $O
# 20s each at 1920x1080, 30fps, and each opens on a 3s fade from black so the
# sampling logic is exercised by the corpus as well as by the codec.
FADE="fade=t=in:st=0:d=3"

# 1. GRAIN — detail plus heavy sensor-like noise. The case AV1 grain synthesis
#    exists for, and the hardest thing to compress honestly.
$FF -hide_banner -loglevel error -f lavfi -i "testsrc2=s=1920x1080:d=20:r=30" \
  -f lavfi -i "nullsrc=s=1920x1080:d=20:r=30,geq=random(1)*255:128:128" \
  -filter_complex "[0:v][1:v]blend=all_mode=average:all_opacity=0.35,$FADE" \
  -c:v libx264 -crf 12 -pix_fmt yuv420p -y $O/grain.mp4

# 2. FLAT ANIMATION — large solid regions, hard edges, little texture. Where
#    per-title should find the most headroom.
$FF -hide_banner -loglevel error -f lavfi -i "color=c=0x1e3a5f:s=1920x1080:d=20:r=30" \
  -vf "drawbox=x=200:y=150:w=700:h=500:color=0xf5d76e:t=fill,drawbox=x=1000:y=400:w=600:h=400:color=0xe8543f:t=fill,drawtext=text='FLAT':fontsize=200:fontcolor=white:x=300:y=800,$FADE" \
  -c:v libx264 -crf 12 -pix_fmt yuv420p -y $O/flat.mp4

# 3. FAST MOTION — high temporal energy, which starves inter prediction.
$FF -hide_banner -loglevel error -f lavfi -i "testsrc2=s=1920x1080:d=20:r=30" \
  -vf "rotate=a=t*3:c=black,zoompan=z='1.4+0.3*sin(in/8)':d=1:s=1920x1080:fps=30,$FADE" \
  -c:v libx264 -crf 12 -pix_fmt yuv420p -y $O/motion.mp4

# 4. DARK — low luma with real detail in it. The content most easily ruined by
#    an over-eager quantiser, and the one a blank-frame check must not reject.
$FF -hide_banner -loglevel error -f lavfi -i "testsrc2=s=1920x1080:d=20:r=30" \
  -vf "curves=all='0/0 0.5/0.09 1/0.22',eq=brightness=-0.06,$FADE" \
  -c:v libx264 -crf 12 -pix_fmt yuv420p -y $O/dark.mp4

for f in $O/*.mp4; do
  printf '%-14s %9s bytes  ' "$(basename $f)" "$(stat -c%s $f)"
  /usr/local/bin/ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0 "$f"
done
tar -C $O -cf /tmp/corpus.tar .

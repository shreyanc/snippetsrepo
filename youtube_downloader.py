import youtube_dl
import csv
import os

class YouTubeDL:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.ydl_opts = {
            'outtmpl': f'{out_dir}/%(id)s.%(ext)s',
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            },
                {'key': 'FFmpegMetadata'}
            ],

        }

    def __call__(self, p):
        """p (pointer) can be link or id"""
        if '/' in p:
            ytid = p.split('/')[-1]
        else:
            ytid = p
        with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([f'http://www.youtube.com/watch?v={ytid}'])


ytd = YouTubeDL(out_dir='/home/shreyan/Downloads/pitchfork_audio')
# ytd('https://www.youtube.com/embed/ZUTu65AXrJw')


def validate_yt_id(sid):
    def check_len(s):
        if len(s) == 11:
            return True
        else:
            return False

    if check_len(sid):
        return sid
    elif check_len(sid.split('?')[0]):
        return sid.split('?')[0]
    elif check_len(sid.strip('"')):
        return sid.strip('"')
    else:
        return False


yt_ids = []
failed_ids = []
with open('yt_links.csv', newline='') as csvfile:
    cr = csv.reader(csvfile)
    for row in cr:
        yid = validate_yt_id(row[1])
        if yid is not False:
            if not os.path.exists(f'/home/shreyan/Downloads/pitchfork_audio/{yid}.mp3'):
                try:
                    ytd(yid)
                except:
                    print(f"Failed to download {row[0]}, {yid}")
                    failed_ids.append(yid)
            else:
                print(yid, 'exists')
        else:
            if len(row[1]) > 0:
                print(row[0], row[1])

print(failed_ids)


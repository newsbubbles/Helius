import mutagen

def get_mp3_metadata(path):
    audio = mutagen.File(path)
    meta = {}
    if audio is not None and audio.tags is not None:
        for k, v in audio.tags.items():
            meta[k] = str(v)
    return meta

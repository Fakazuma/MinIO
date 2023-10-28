import os
import glob


def upload_files_to_s3(s3client, src_dir: str, s3_trg_dir: str, bucket: str) -> None:
    files = glob.glob(os.path.join(src_dir, '**', '*.*'), recursive=True)
    for file in files:
        s3fname = file.split(src_dir)[1][1:]  # bad
        s3client.Bucket(bucket).upload_file(file, os.path.join(s3_trg_dir, s3fname))


def download_files_from_s3(s3client, bucket: str,  s3_src_file: str, trg_dir: str = '.cache') -> str:
    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)
    fpath = os.path.join(trg_dir, os.path.basename(s3_src_file))
    s3client.Bucket(bucket).download_file(s3_src_file, fpath)
    return fpath

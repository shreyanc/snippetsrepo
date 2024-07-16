import os

import numpy as np
import tensorflow as tf
from pydub import AudioSegment
from glob import glob
from tqdm import tqdm

from musika.utils import Utils_functions

class MusikaEncoderDecoder:
    def __init__(self, args, models_ls):

        self.args = args
        self.U = Utils_functions(args)

        (
            critic,
            gen,
            self.enc,
            self.dec,
            self.enc2,
            self.dec2,
            gen_ema,
            [opt_dec, opt_disc],
            self.switch,
        ) = models_ls

    def encode_audio(self, audio_wf):
        c = 0
        time_compression_ratio = 16  # TODO: infer time compression ratio
        shape2 = self.args.shape

        wv = audio_wf.T  # shape = [samples, channels]

        if (
            wv.shape[0]
            > self.args.hop * self.args.shape * 2 + 3 * self.args.hop
        ):

            rem = (wv.shape[0] - (3 * self.args.hop)) % (
                self.args.shape * self.args.hop
            )

            if rem != 0:
                wv = tf.concat([wv, tf.zeros([rem, 2], dtype=tf.float32)], 0)

            chls = []
            for channel in range(2):

                x = wv[:, channel]
                x = tf.expand_dims(
                    tf.transpose(
                        self.U.wv2spec(x, hop_size=self.args.hop), (1, 0)
                    ),
                    -1,
                )
                ds = []
                num = x.shape[1] // self.args.shape
                rn = 0
                for i in range(num):
                    ds.append(
                        x[
                            :,
                            rn
                            + (i * self.args.shape) : rn
                            + (i * self.args.shape)
                            + self.args.shape,
                            :,
                        ]
                    )
                del x
                ds = tf.convert_to_tensor(ds, dtype=tf.float32)
                lat = self.U.distribute_enc(ds, self.enc)
                del ds
                lat = tf.split(lat, lat.shape[0], 0)
                lat = tf.concat(lat, -2)
                lat = tf.squeeze(lat)

                ds2 = []
                num2 = lat.shape[-2] // shape2
                rn2 = 0
                for j in range(num2):
                    ds2.append(
                        lat[rn2 + (j * shape2) : rn2 + (j * shape2) + shape2, :]
                    )
                ds2 = tf.convert_to_tensor(ds2, dtype=tf.float32)
                lat = self.U.distribute_enc(tf.expand_dims(ds2, -3), self.enc2)
                del ds2
                lat = tf.split(lat, lat.shape[0], 0)
                lat = tf.concat(lat, -2)
                lat = tf.squeeze(lat)
                chls.append(lat)

            lat = tf.concat(chls, -1)

            del chls
            
            return lat

    def decode_audio(self, lat):
        lat = tf.expand_dims(lat, 0)
        lat = tf.expand_dims(lat, 0)
        wv = self.U.decode_waveform(lat, self.dec, self.dec2)
        return wv.T


if __name__ == "__main__":

    # parse args
    args = parse_args()

    # initialize networks
    M = Models_functions(args)
    M.download_networks()
    models_ls = M.get_networks()

    # encode samples
    # U = UtilsEncode_functions(args)
    # if args.whole:
    #     U.compress_whole_files(models_ls)
    # else:
    #     U.compress_files(models_ls)

    musika = MusikaEncoderDecoder(args, models_ls)

    ds = GMD10s('/home/shreyan/Desktop/test_audio', ret_filename=True)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    zs = []
    for x, f in tqdm(dl, total=len(dl)):
        x = x.to(device)
        z = musika.encode_audio(x.squeeze()) 
        print(z.shape, f)

        signal = AudioSignal(musika.decode_audio(z), sample_rate=44100)
        signal.write('recon/recon_'+f[0])

        print('saved reconstruction for', f[0])

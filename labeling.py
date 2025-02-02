import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from vggish_input import waveform_to_examples
from vggish_postprocess import Postprocessor
import vggish_slim

tf.compat.v1.disable_eager_execution()

def define_model():
    vggish_slim.define_vggish_slim(training=False)

define_model()

def load_audio_and_extract_mel_spectrogram(file_path):
    try:
        waveform, sr = librosa.load(file_path, sr=16000, mono=True)
        mel_spectrogram = waveform_to_examples(waveform, sr)
        return mel_spectrogram
    except Exception as e:
        print(f"Erro ao carregar áudio {file_path}: {e}")
        return None

def extract_features_vggish(mel_spectrogram, model_path='vggish_model.ckpt', pca_path='vggish_pca_params.npz'):
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        try:
            saver.restore(sess, model_path)
            print("Modelo VGGish restaurado com sucesso!")
        except tf.errors.NotFoundError as e:
            print(f"Erro ao restaurar o modelo VGGish: {e}")
            return None

        features_tensor = sess.graph.get_tensor_by_name('vggish/input_features:0')
        embedding_tensor = sess.graph.get_tensor_by_name('vggish/embedding:0')
        [embedding] = sess.run([embedding_tensor], feed_dict={features_tensor: mel_spectrogram})

        pproc = Postprocessor(pca_path)
        embedding = pproc.postprocess(embedding)

    return embedding

def process_jamendo_audio_with_genre(audio_directory, output_dir, annotations_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotations = pd.read_csv(annotations_file, sep='\t')
    annotations['TRACK_ID'] = annotations['TRACK_ID'].apply(lambda x: x.split('_')[-1].zfill(7))

    for root, _, files in os.walk(audio_directory):
        for file_name in files:
            if file_name.endswith(".mp3") or file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                print(f"Processando arquivo: {file_name}")

                mel_spectrogram = load_audio_and_extract_mel_spectrogram(file_path)
                if mel_spectrogram is not None:
                    embeddings = extract_features_vggish(mel_spectrogram)
                    
                    if embeddings is not None:
                        # Extrair a parte numérica do nome do arquivo, removendo o sufixo ".low"
                        track_id = os.path.splitext(file_name)[0].split('.')[0].zfill(7)
                        genre_row = annotations.loc[annotations['TRACK_ID'] == track_id, 'LABELS']
                        
                        if not genre_row.empty:
                            genre_label = genre_row.values[0].replace('genre:', '')
                            if genre_label:
                                output_data = {
                                    "embeddings": embeddings,
                                    "genre": genre_label
                                }
                                np.save(os.path.join(output_dir, f"{track_id}_vggish.npy"), output_data)
                                print(f"Features e gênero '{genre_label}' salvos para {file_name}")
                            else:
                                print(f"Gênero não identificado para o arquivo {file_name}.")
                        else:
                            print(f"ID {track_id} não encontrado nas anotações.")
                    else:
                        print(f"Erro: Embeddings não extraídos para {file_name}")
                else:
                    print(f"Erro: Espectrograma de Mel não extraído para {file_name}")

audio_directory = '/home/missantroop/SongAnalysis2/mtg-jamendo-dataset/downloaded-data'
output_dir = '/home/missantroop/SongAnalysis2/mtg-jamendo-dataset/songresult'
annotations_file = '/home/missantroop/SongAnalysis2/models/research/audioset/vggish/labeled_filtered_file.tsv'

process_jamendo_audio_with_genre(audio_directory, output_dir, annotations_file)

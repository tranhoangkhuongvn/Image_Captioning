import tensorflow as tf
import streamlit as st
# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import collections
import random
import re
import numpy as np
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
import time
import json
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm
import cv2


from transformer import *
from base_model import *



##### HELPER FUNCTIONS ######
# Load and define the image extractor
# image_name ---> feature vectors (8x8x2048)
def init_transformer_model():
    
    opt = {
    'train_src_data':'./data/train.en',
    'train_trg_data':'./data/train.vi',
    'valid_src_data':'./data/tst2013.en',
    'valid_trg_data':'./data/tst2013.vi',
    'src_lang':'en_core_web_sm',
    'trg_lang':'en_core_web_sm',#'vi_spacy_model',
    'max_strlen':160,
    'batchsize':1500,
    'device':'cpu',
    'd_model': 512,
    'n_layers': 6,
    'heads': 8,
    'dropout': 0.1,
    'lr':0.0001,
    'epochs':30,
    'printevery': 200,
    'k':5,
    }   
    os.makedirs('./data/', exist_ok=True)
    train_src_data, train_trg_data = read_data(opt['train_src_data'], opt['train_trg_data'])
    valid_src_data, valid_trg_data = read_data(opt['valid_src_data'], opt['valid_trg_data'])

    SRC, TRG = create_fields(opt['src_lang'], opt['trg_lang'])
    train_iter = create_dataset(train_src_data, train_trg_data, opt['max_strlen'], opt['batchsize'], opt['device'], SRC, TRG, istrain=True)
    valid_iter = create_dataset(valid_src_data, valid_trg_data, opt['max_strlen'], opt['batchsize'], opt['device'], SRC, TRG, istrain=False)

    src_pad = SRC.vocab.stoi['<pad>']
    trg_pad = TRG.vocab.stoi['<pad>']
    model = Transformer(len(SRC.vocab), len(TRG.vocab), opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout'])

    model.load_state_dict(torch.load('./my_transformer.pt', map_location='cpu') )


    return model, SRC, TRG, opt



@st.cache(suppress_st_warning=True)
def load_image_extractor(arg=None):
    print('load image extractor')
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input

    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    return image_features_extract_model


def evaluate(image=None, image_path=None, image_features_extract_model=None, model_checkpoint=None):
    if image == None and image_path == None:
        return "Error evaluating!"
    
    encoder = model_checkpoint.encoder
    decoder = model_checkpoint.decoder
    optimizer = model_checkpoint.optimizer
    max_length = params['max_length']
    attention_features_shape = params['attention_features_shape']


    # loading
    with open('final_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    if image:
        temp_input = tf.expand_dims(load_image(image)[0], 0)
    else:
        temp_input = tf.expand_dims(load_image(image_path)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    print('passed')
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image=None, image_path=None, result=None, attention_plot=None):
    if image_path:
        temp_image = np.array(Image.open(image_path))
    else:
        temp_image = np.array(image)

    fig = plt.figure(figsize=(20, 20))

    len_result = len(result)
    print(len_result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (12, 12))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    fig.tight_layout()
    #plt.show()
    st.pyplot(fig)


def load_image(image_path):
    if isinstance(image_path, str):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
    else:    
        img = image_path
        img = np.array(img)
    
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    
    return img, image_path



###### END HELPER ########

# Load the tensorflow model for Image Captioning
@st.cache(suppress_st_warning=True)
def load_annotation_file():
    print('load annotation file')
    #annotation_file_train = './annotations/captions_train2014.json'
    annotation_file_val = './annotations/captions_val2014.json'

    #image_folder_train = '/train2014/'
    image_folder_val = '/val2014/'
    #image_folder_test = '/test2014/'

    #PATH_TRAIN = os.path.abspath('.') + image_folder_train
    PATH_VAL = os.path.abspath('.') + image_folder_val
    #PATH_TEST = os.path.abspath('.') + image_folder_test


    with open(annotation_file_val, 'r') as f:
        annotations_val = json.load(f)

    image_path_to_caption_val = collections.defaultdict(list)

    for value in tqdm( annotations_val['annotations']):
        caption = f"<start> {value['caption']} <end>"
        image_path = PATH_VAL + 'COCO_val2014_' + '%012d.jpg' % (value['image_id'])
        image_path_to_caption_val[image_path].append(caption)

    return image_path_to_caption_val

@st.cache(suppress_st_warning=True)
def init_params():
    print('init params')
    params = {}
    params['top_k'] = 10000
    params['batch_size']= 32
    params['buffer_size'] = 1000
    params['embedding_dim'] = 1024
    params['units'] = 1024
    params['vocab_size'] = params['top_k'] + 1
    params['max_length'] = 52
    params['features_shape'] = 2048
    params['attention_features_shape'] = 64

    return params




# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape

def load_model_from_checkpoint(params):
    print('load model from checkpoint')
    embedding_dim = params['embedding_dim']
    units = params['units']
    vocab_size = params['vocab_size']
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()

    checkpoint_dir = './checkpoints/train_eng_v3/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
                                    optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder,
                                    )

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    return checkpoint


# Load the pytorch model for eng-viet translation

# display the image with the english caption
# captions on the validation set
# rid = np.random.randint(0, len(img_name_val))
# image = img_name_val[rid]
# real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
# result, attention_plot = evaluate(image)

# print ('Real Caption:', real_caption)
# print ('Prediction Caption:', ' '.join(result))
# plot_attention(image, result, attention_plot)
# display the vietnamese caption

# show the attention visualisation

    


if __name__ == '__main__':
    st.title('Automatic Image Captioning with Attention')

    st.write("""
    ### You have 2 options: randomly select a test image or upload your own image
    """)

    params = init_params()
    
    image_path_to_caption_val = load_annotation_file()

    image_features_extract_model = load_image_extractor()
    transformer_model, SRC, TRG, opt = init_transformer_model()

    # Test button 
    count = 0
    random_image = st.button('New random image')
    if random_image:
        ckpt = load_model_from_checkpoint(params)
        count += 1
        st.write('New image {}'.format(count))
        image_list = list( image_path_to_caption_val.keys() )
        random_index = np.random.randint(0, len(image_path_to_caption_val.keys()))
        image_path = image_list[random_index]
        print(image_path)
        image = Image.open(image_path)
        real_caption = image_path_to_caption_val[image_path][0]
        st.image(image, caption=real_caption, use_column_width=True)
        

        result, attention_plot = evaluate(image_path=image_path, image_features_extract_model=image_features_extract_model, model_checkpoint=ckpt)

        #print ('Real Caption:', real_caption)
        predicted_captions = ' '.join(result[:-1])
        print ('Prediction Caption:', ' '.join(result[:-1]))
        st.write('English Caption:', ' '.join(result[:-1]))
        
        #sentence='My family was not poor , and myself , I had never experienced hunger .'
        trans_sent = translate_sentence(predicted_captions, transformer_model, SRC, TRG, opt['device'], opt['k'], opt['max_strlen'])
        st.write("Vietnamese caption:", trans_sent)


        plot_attention(image_path=image_path, result=result, attention_plot=attention_plot)


    upload_button = st.button('Upload your image')
    if upload_button:
        st.subheader("Images")
        uploaded_file = st.file_uploader("Upload Files", type=["png","jpeg", "jpg"])
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "Type": uploaded_file.type,
                "Size": uploaded_file.size,
            }
            ckpt = load_model_from_checkpoint(params)
            st.write(file_details)
            image = Image.open(uploaded_file)
            print(type(image))
            #st.image(img) #, width=250, height=250)

            #model = load_model('data/flower_model.h5')
            #data = uploaded_file.read()
            #class_name = preprocess_predict(model, data)
            #message = "This image is likely to be "+class_name
            #st.success(message)
            #real_caption = image_path_to_caption_val[image_path][0]
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            result, attention_plot = evaluate(image =image, image_features_extract_model=image_features_extract_model, model_checkpoint=ckpt)
            
            predicted_captions = ' '.join(result[:-1])
            #print ('Real Caption:', real_caption)
            print ('Prediction Caption:', ' '.join(result[:-1]))
            st.write('English Caption:', ' '.join(result[:-1]))

            trans_sent = translate_sentence(predicted_captions, transformer_model, SRC, TRG, opt['device'], opt['k'], opt['max_strlen'])
            st.write("Vietnamese caption:", trans_sent)
            plot_attention(image=image, result=result, attention_plot=attention_plot)
            st.success("Done")
    




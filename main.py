import dataset
import Emotion
import numpy as np
import os, sys
import csv
import get_image_from_camera
import cv2
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", help="evaluate on a image input from path specified or camera, should be used only when training is done and value should be true")
    parser.add_argument("--image", help="value must be 'path-to-img-file' or 'webcam'")
    parser.add_argument("--train", help="trains from the dataset in the path: 'current_directory/dataset/fer2013.csv'")
    parser.add_argument("--checkpoint_directory", help="path to checkpoint directory")
    parser.add_argument("--steps", help="Number of iterations of training to be done, default is 20000", type=int)
    args = parser.parse_args()

    emotion_cnn = Emotion.emotion(args.checkpoint_directory, args.steps)

    if args.train == "true":
        relearn = False

        current_dir = r"C:\Users\Joy.DESKTOP-M53NCFS\Documents\GitHub\Emotion-recognizer"
        train_file = current_dir+"\\dataset\\train.txt"
        valid_file = current_dir+"\\dataset\\valid.txt"
        test_file = current_dir+"\\dataset\\test.txt"

        obj = dataset.dataset_reader()


        print ('\n')

        for checkpoint_path, checkpoint_name, checkpoint_files in os.walk(current_dir+"\checkpoints"):
            if checkpoint_files:
                relearn = True

        if not relearn:
            filename = current_dir+"\\dataset\\fer2013.csv"



            total_data = obj.get_dataset_before_learn(filename)
            shuffled_data = obj.shuffle_dataset(total_data)

            train_data, train_label, valid_data, valid_label, test_data ,test_label = obj.split_dataset(shuffled_data)
            print (train_label.shape, " ", valid_data.shape, " ", test_data.shape)

            train_data = obj.normalize_dataset(train_data)
            valid_data = obj.normalize_dataset(valid_data)
            test_data = obj.normalize_dataset(test_data)

            print (type(train_data))

            with open(train_file, 'wb') as f:
                print ("Creating .txt file for training data ...")
                np.savetxt(f, np.c_[train_label,  train_data], fmt = '%.7f')
                print ("done")

            with open(valid_file, 'wb') as f:
                print ("Creating .txt file for validating data ...")
                np.savetxt(f, np.c_[valid_label,  valid_data], fmt = '%.7f')
                print ("done")

            with open(test_file, 'wb') as f:
                print ("Creating .txt file for testing data ...")
                np.savetxt(f, np.c_[test_label,  test_data], fmt = '%.7f')
                print ("done")

            emotion_cnn.preprocess_datasets(train_data, train_label, valid_data, valid_label, test_data, test_label)
            emotion_cnn.train_test_validate()

        else:
            train_data, train_label, valid_data, valid_label, test_data ,test_label = obj.get_dataset_during_learn(train_file, valid_file, test_file)
            emotion_cnn.preprocess_datasets(train_data, train_label, valid_data, valid_label, test_data, test_label)
            emotion_cnn.train_test_validate()

    elif args.evaluate == "true":
        if args.image =="webcam":
            print ("\n\nPress 'esc' to exit webcam and only after 'face detected' is printed (Else error will occur as no image will be fed into neural network)")
            img = get_image_from_camera.get_image()
            if not img:
                print ("No image captured... improve light quality and press 'esc' after seeing 'face detected' message")
                sys.exit(0)
        else:
            path_to_img = args.image
            image = cv2.imread(path_to_img, 0)

            image = image.astype(np.float32)
            temp = image.copy()
            temp.fill(128)

            img = (image-temp)/128
            img = cv2.resize(img, (48,48))


        print ("Most probable emotion is : ", emotion_cnn.evaluate(img))


if __name__ == '__main__':
    main()

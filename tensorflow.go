package main

/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
import (
	"bufio"
	"io/ioutil"
	"log"
	"os"
	"encoding/base64"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type item struct {
	Name    string
	Precent int
}

var modelfile string = "path/output_graph.pb"
var labelsfile string = "path/output_labels.txt"
var sess *tf.Session
var graph *tf.Graph

func init() {
	model, err := ioutil.ReadFile(modelfile)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph = tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a sess for inference over graph.
	sess, err = tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
}

func predict(img string) []item {
	// An example for using the TensorFlow Go API for image recognition
	// using a pre-trained inception model (http://arxiv.org/abs/1512.00567).
	//
	// Sample usage: <program> -dir=/tmp/modeldir -image=/path/to/some/jpeg
	//
	// The pre-trained model takes input in the form of a 4-dimensional
	// tensor with shape [ BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3 ],
	// where:
	// - BATCH_SIZE allows for inference of multiple images in one pass through the graph
	// - IMAGE_HEIGHT is the height of the images on which the model was trained
	// - IMAGE_WIDTH is the width of the images on which the model was trained
	// - 3 is the (R, G, B) values of the pixel colors represented as a float.
	//
	// And produces as output a vector with shape [ NUM_LABELS ].
	// output[i] is the probability that the input image was recognized as
	// having the i-th label.
	//
	// A separate file contains a list of string labels corresponding to the
	// integer indices of the output.
	//
	// This example:
	// - Loads the serialized representation of the pre-trained model into a Graph
	// - Creates a Session to execute operations on the Graph
	// - Converts an image file to a Tensor to provide as input to a Session run
	// - Executes the Session and prints out the label with the highest probability
	//
	// To convert an image file to a Tensor suitable for input to the Inception model,
	// this example:
	// - Constructs another TensorFlow graph to normalize the image into a
	//   form suitable for the model (for example, resizing the image)
	// - Creates an executes a Session to obtain a Tensor in this normalized form.

	// Run inference on *imageFile.
	// For multiple images, session.Run() can be called in a loop (and
	// concurrently). Alternatively, images can be batched since the model
	// accepts batches of image data as input.
	tensor, err := makeTensorFromImage(img)
	if err != nil {
		log.Fatal(err)
	}
	output, err := sess.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("DecodeJpeg/contents").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("final_result").Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}
	// output[0].Value() is a vector containing probabilities of
	// labels for each image in the "batch". The batch size was 1.
	// Find the most probably label index.
	probabilities := output[0].Value().([][]float32)[0]
	return printBestLabel(probabilities, labelsfile)
}

func printBestLabel(probabilities []float32, labelsFile string) (name []item) {
	idx := [5]int{}
	for i, p := range probabilities {
		if p > probabilities[idx[0]] {
			idx = [5]int{i, idx[0], idx[1], idx[2], idx[3]}
		}
	}
	// Found the best match. Read the string from labelsFile, which
	// contains one line per label.
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
	name = []item{
		{labels[idx[0]], int(probabilities[idx[0]]*100)},
		{labels[idx[1]], int(probabilities[idx[1]]*100)},
		{labels[idx[2]], int(probabilities[idx[2]]*100)},
		{labels[idx[3]], int(probabilities[idx[3]]*100)},
		{labels[idx[4]], int(probabilities[idx[4]]*100)},
	}
	return name
}

// Convert the image in filename to a Tensor suitable as input to the Inception model.
func makeTensorFromImage(base64Img string) (*tf.Tensor, error) {
	bytes, err := base64.StdEncoding.DecodeString(base64Img)
	if err != nil {
		return nil, err
	}
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		return nil, err
	}
	return tensor, nil
}

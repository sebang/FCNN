using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FCNN
{
    class NeuralNetwork
    {
        public int num_input_;  // the number of input layer neuron
        public int num_output_; // the number of output layer neuron
        public int num_all_layers_; // hidden layers + 2

        public float bias_;   // constant bias
        public float alpha_;  // learning rate

        List<List<float>> layer_neuron_act_ = new List<List<float>> ();  // this includes bias.
        List<List<float>> layer_neuron_grad_ = new List<List<float>> (); // this inclues bias.
        List<MatrixMN> weights_ = new List<MatrixMN> ();

        List<int> num_layer_acts_ = new List<int> (); // this includes bias.

        public NeuralNetwork (int _num_input, int _num_output, int _num_hidden_layers)
        {
            initialize (_num_input, _num_output, _num_hidden_layers);
        }

        private void initialize (int _num_input, int _num_output, int _num_hidden_layers)
        {
            num_input_ = _num_input;
            num_output_ = _num_output;
            num_all_layers_ = _num_hidden_layers + 2;

            // set the number of each layer's act.
            num_layer_acts_.Clear ();

            num_layer_acts_.Add (num_input_ + 1); // input layer. +1 is for bias.

            for (int l = 1; l < _num_hidden_layers + 1; l++) // hidden layers.
                num_layer_acts_.Add (num_input_ + 1); // default value. +1 is for bias.

            num_layer_acts_.Add (num_output_ + 1); // output layer. +1 is for bias.

            // set bias and alpha.
            bias_ = 1;
            alpha_ = 0.15f;

            // create act storage of layers neuron.
            layer_neuron_act_.Clear ();
            for (int l = 0; l < num_all_layers_; l++) // l = layer.
            {
                layer_neuron_act_.Add (new List<float> ());
         
                for (int d = 0; d < num_layer_acts_ [l]; d++) // d = dimension.
                    layer_neuron_act_ [l].Add (0.0f);
                layer_neuron_act_ [l] [num_layer_acts_ [l] - 1] = bias_;
            }

            // create gradient storage of layers neuron.
            layer_neuron_grad_.Clear ();
            for (int l = 0; l < num_all_layers_; l++)
            {
                layer_neuron_grad_.Add (new List<float> ());

                for (int d = 0; d < num_layer_acts_ [l]; d++)
                    layer_neuron_grad_ [l].Add (0.0f);
            }

            // create weight matrix between layers.
            weights_.Clear ();
            for (int l = 0; l < num_all_layers_ - 1; l++) // -1: between layers.
            {
                // create matrix between (i + 1) and (i).
                // row x column = (dimension of next layer -1 for bias) x (dimension of prev layer - this includes bias)
                weights_.Add (new MatrixMN ());
                weights_ [l].initialize (layer_neuron_act_[l + 1].Count () - 1, layer_neuron_act_[l].Count ());

                // random initialization.
                // caution: I guess C# Random.Next include srand(time(NULL));
                Random rand = new Random ();
                for (int ix = 0; ix < weights_ [l].num_ix_; ix++)
                    weights_ [l].values_ [ix] = ((float)rand.Next ()/Int32.MaxValue) * 0.1f;
                 // weights_ [l].values_ [ix] = (float)rand.NextDouble() * 0.1f;
            }
        }

        public void setInputVector (List<float> input)
        {
            // use num_input_ in case input vector doesn't include bias

            if(input.Count () < num_input_)
            {
                Console.Write ("Input dimension is wrong.\n");
                return;
            }

            for (int d = 0; d < num_input_; d++)
                layer_neuron_act_ [0] [d] = input [d];
        }

        public void propForward ()
        {
            for (int l = 0; l < weights_.Count (); l++) // number of connections
            {
                // multiply
                List<float> test = new List<float> ();
                layer_neuron_act_ [l + 1] = weights_ [l].multiply (layer_neuron_act_ [l], layer_neuron_act_ [l + 1]);

                // activate
                // The last component of layer_neuron_act_[l + 1], bias, shouldn't be updated.
                layer_neuron_act_ [l + 1] = applyRELUToVector (layer_neuron_act_ [l + 1]);
            }
        }

        public void copyOutputVector (out List<float> copy, bool copy_bias = false)
        {
            copy = new List<float> ();
            int output_layer = layer_neuron_act_.Count () - 1;
            int num_output;

            if (copy_bias == false)
                num_output = num_output_;
            else
                num_output = num_output_ + 1;

            for (int d = 0; d < num_output; d++)
                copy.Add (layer_neuron_act_ [output_layer] [d]);
        }

        public void propBackward (List<float> target)
        {
            // calculate gradients of output layer (get error for backward)
            int output_layer = layer_neuron_grad_.Count () - 1;

            for (int d = 0; d < layer_neuron_grad_[output_layer].Count () - 1; d++) // skip last component (bias)
            {
                float output_value = layer_neuron_act_ [output_layer] [d];

                // gradient 
                // = delta E / delta w 
                // = (y0 - y0,target) * (delta f(x) / delta sigma(x) = getRELUGradFromY)
                layer_neuron_grad_ [output_layer] [d] = (target [d] - output_value) * getRELUGradFromY(output_value);
            }

            // calculate gradients of hidden layers. (backward error)
            for (int l = weights_.Count () - 1; l >= 0; l--)
            {
                // transpose for backward propagation
                layer_neuron_grad_ [l] = weights_ [l].multiplyTransposed (layer_neuron_grad_ [l + 1], layer_neuron_grad_ [l]);

                for (int d = 0; d < layer_neuron_act_ [l].Count () - 1; d++)
                    layer_neuron_grad_ [l] [d] *= getRELUGradFromY (layer_neuron_act_ [l] [d]);
            }

            // update weights after all gradients are calculated
            // w_updated = w - alpha * (delta E / delta w)
            for (int l = weights_.Count () - 1; l >= 0; l--)
                weights_[l] = updateWeight (weights_[l], layer_neuron_grad_[l + 1], layer_neuron_act_[l]);

        }

        MatrixMN updateWeight (MatrixMN weight_matrix, List<float> next_layer_grad, List<float> prev_layer_act)
        {
            for (int row = 0; row < weight_matrix.num_rows_; row++)
            {
                for (int col = 0; col < weight_matrix.num_cols_; col++)
                {
                    float delta_w = alpha_ * next_layer_grad [row] * prev_layer_act [col];
                    int index_1d = weight_matrix.get1DIndex (row, col);

                    weight_matrix.values_ [index_1d] += delta_w;
                }
            }
            return weight_matrix;
        }

        List<float> applyRELUToVector (List<float> vector)
        {
            for (int d = 0; d < vector.Count () - 1; d++) // don't apply activation function to bias
                vector [d] = Math.Max(0.0f, vector[d]);

            return vector;
        }

        float getRELUGradFromY (float x)
        {
            if (x > 0.0f) return 1.0f;
            else return 0.0f;
        }
    }
}

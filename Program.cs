using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FCNN
{
    class Program
    {
        static void Main (string [] args)
        {
            List<float> x = new List<float> () { 0.0f, 0.0f };  // input
            List<float> y_target = new List<float> () { 0.3f }; // desire output
            List<float> y_temp;

            NeuralNetwork nn_ = new NeuralNetwork (2, 1, 1); // input, output, hidden
            nn_.alpha_ = 0.1f;

            for (int t = 0; t < 30; t++)
            {
                nn_.setInputVector (x);
                nn_.propForward ();

                nn_.copyOutputVector (out y_temp);
                Console.Write (y_temp [0] + "\n");

                nn_.propBackward (y_target);
            }

            // XOR 연산은 잘 작동하지 않습니다. (될 때도 있고 안될 때도 있고...)
            //List<Xor> xors = new List<Xor> ();

            //xors.Add (new Xor (0.0f, 0.0f, 0.0f));
            //xors.Add (new Xor (1.0f, 0.0f, 1.0f));
            //xors.Add (new Xor (0.0f, 1.0f, 1.0f));
            //xors.Add (new Xor (1.0f, 1.0f, 0.0f));

            //NeuralNetwork nn_ = new NeuralNetwork (2, 1, 1); // input, output, hidden
            //nn_.alpha_ = 0.1f;

            //for (int t = 0; t < 1000; t++) // t = training.
            //{
            //    foreach (Xor xor in xors)
            //    {
            //        nn_.setInputVector (xor.x_);
            //        nn_.propForward ();

            //        List<float> y_temp;
            //        nn_.copyOutputVector (out y_temp);
            //        Console.Write ("x0(" + xor.x_ [0] +
            //                      ") x1(" + xor.x_ [1] +
            //                      ") now y(" + y_temp [0] +
            //                      ") desire y(" + xor.y_ [0] + ")\n");

            //        nn_.propBackward (xor.y_);
            //    }

            //    Console.Write ("\n");
            //}
        }
    }

    //class Xor
    //{
    //    public List<float> x_ = new List<float> ();
    //    public List<float> y_ = new List<float> ();

    //    public Xor (float _x0, float _x1, float _y)
    //    {
    //        x_.Clear (); y_.Clear ();

    //        x_.Add (_x0);
    //        x_.Add (_x1);
    //        y_.Add (_y);
    //    }
    //}
}
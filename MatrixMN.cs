using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace FCNN
{
    class MatrixMN
    {
        public int num_rows_; // m_
        public int num_cols_; // n_
        public int num_ix_ { get { return num_rows_ * num_cols_; } }
        public List<float> values_;

        public MatrixMN ()
        { values_ = new List<float> ();  num_rows_ = 0; num_cols_ = 0; }

        public void initialize (int _m, int _n)
        {
            values_.Clear ();
            num_rows_ = _m;
            num_cols_ = _n;

            int num_all = num_rows_ * num_cols_;

            for (int i = 0; i < num_all; i++)
                values_.Add (0f);
        }

        public List<float> multiply (List<float> vector, List<float> result)
        {
            Debug.Assert (num_rows_ <= result.Count ());
            Debug.Assert (num_cols_ <= vector.Count ());

            for (int row = 0; row < num_rows_; row++)
            {
                int ix = row * num_cols_;

                result [row] = 0f;
                float temp;
                for (int col = 0; col < num_cols_; col++, ix++)
                {
                    temp = values_ [ix] * vector [col];
                    result [row] += temp;
                }
            }

            return result;
        }

        public List<float> multiplyTransposed (List<float> vector, List<float> result)
        {
            Debug.Assert (num_rows_ <= vector.Count ());
            Debug.Assert (num_cols_ <= result.Count ());

            for (int col = 0; col < num_cols_; col++)
            {
                result [col] = 0f;

                for (int row = 0, ix = col; row < num_rows_; row++, ix += num_cols_)
                    result [col] += values_ [ix] * vector [row];
            }

            return result;

            //Note: You may transpose matrix and then multiply for better performance.
            //See Eigen library. http://eigen.tuxfamily.org/index.php?title=Main_Page
        }

        public int get1DIndex (int row, int column)
        {
            Debug.Assert (row >= 0);
            Debug.Assert (column >= 0);
            Debug.Assert (row < num_rows_);
            Debug.Assert (row < num_cols_);

            return column + row * num_cols_;
        }
    }
}

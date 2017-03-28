<?php
namespace ConvNet;

class RegressionLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        $this->num_inputs = $opt['in_sx'] * $opt['in_sy'] * $opt['in_depth'];
        $this->out_depth = $this->num_inputs;
        $this->out_sx = 1;
        $this->out_sy = 1;
        $this->layer_type = 'regression';
    }

    public function forward($V, $is_training)
    {
        $this->in_act = $V;
        $this->out_act = $V;
        return $V; // identity function
    }

    public function backward($y = null)
    {
        // compute and accumulate gradient wrt weights and bias of this layer
        $x = $this->in_act;
        $x->dw = array_fill(0, count($x->w), 0); // zero out the gradient of input Vol
        $loss = 0.0;

        if (is_array($y)) {
            for ($i = 0; $i < $this->out_depth; $i++) {
                $dy = $x->w[$i] - $y[$i];
                $x->dw[$i] = $dy;
                $loss += 0.5 * $dy * $dy;
            }
        } else if (is_numeric($y)) {
            // lets hope that only one number is being regressed
            $dy = $x->w[0] - $y;
            $x->dw[0] = $dy;
            $loss += 0.5 * $dy * $dy;
        } else {
            // assume it is a struct with entries .dim and .val
            // and we pass gradient only along dimension dim to be equal to val
            $i = $y->dim;
            $yi = $y->val;
            $dy = $x->w[$i] - $yi;
            $x->dw[$i] = $dy;
            $loss += 0.5 * $dy * $dy;
        }

        return $loss;
    }
}

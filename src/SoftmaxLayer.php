<?php
namespace ConvNet;

class SoftmaxLayer extends Layer
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
        $this->layer_type = 'softmax';
    }

    public function forward($V, $is_training)
    {
        $this->in_act = $V;

        $A = new Vol(1, 1, $this->out_depth, 0.0);

        // compute max activation
        $as = $V->w;
        $amax = $V->w[0];

        for ($i = 1; $i < $this->out_depth; $i++) {
            if ($as[$i] > $amax) {
                $amax = $as[$i];
            }
        }

        // compute exponentials (carefully to not blow up)
        $es = array_fill(0, $this->out_depth, 0);
        $esum = 0.0;

        for ($i = 0; $i < $this->out_depth; $i++) {
            $e = exp($as[$i] - $amax);
            $esum += $e;
            $es[$i] = $e;
        }

        // normalize and output to sum to one
        for ($i = 0; $i < $this->out_depth; $i++) {
            $es[$i] /= $esum;
            $A->w[$i] = $es[$i];
        }

        $this->es = $es; // save these for backprop
        $this->out_act = $A;

        return $this->out_act;
    }

    public function backward($y = null)
    {
        // compute and accumulate gradient wrt weights and bias of this layer
        $x = $this->in_act;
        $x->dw = array_fill(0, count($x->w), 0); // zero out the gradient of input Vol

        for ($i = 0; $i < $this->out_depth; $i++) {
            $indicator = $i === $y ? 1.0 : 0.0;
            $mul = -($indicator - $this->es[$i]);
            $x->dw[$i] = $mul;
        }

        // loss is the class negative log likelihood
        return -log($this->es[$y]);
    }
}

<?php
namespace ConvNet;

class MaxoutLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        $this->out_sx = $opt['in_sx'];
        $this->out_sy = $opt['in_sy'];
        $this->out_depth = floor($opt['in_depth'] / $this->group_size);
        $this->layer_type = 'maxout';

        $this->switches = array_fill(0, $this->out_sx * $this->out_sy * $this->out_depth, 0); // useful for backprop
    }

    public function forward($V, $is_training)
    {
        $this->in_act = $V;
        $N = $this->out_depth;
        $V2 = new Vol($this->out_sx, $this->out_sy, $this->out_depth, 0.0);

        // optimization branch. If we're operating on 1D arrays we dont have
        // to worry about keeping track of x,y,d coordinates inside
        // input volumes. In convnets we do :(
        if ($this->out_sx === 1 && $this->out_sy === 1) {
            for ($i = 0; $i < $N; $i++) {
                $ix = $i * $this->group_size; // base index offset
                $a = $V->w[$ix];
                $ai = 0;

                for ($j = 1; $j < $this->group_size; $j++) {
                    $a2 = $V->w[$ix + $j];

                    if ($a2 > $a) {
                        $a = $a2;
                        $ai = $j;
                    }
                }

                $V2->w[$i] = $a;
                $this->switches[$i] = $ix + $ai;
            }
        } else {
            $n = 0; // counter for switches
            for ($x = 0; $x < $V->sx; $x++) {
                for ($y = 0; $y < $V->sy; $y++) {
                    for ($i = 0; $i < $N; $i++) {
                        $ix = $i * $this->group_size;
                        $a = $V->get($x, $y, $ix);
                        $ai = 0;

                        for ($j = 1; $j < $this->group_size; $j++) {
                            $a2 = $V->get($x, $y, $ix + $j);
                            if ($a2 > $a) {
                                $a = $a2;
                                $ai = $j;
                            }
                        }

                        $V2->set($x, $y, $i, $a);
                        $this->switches[$n] = $ix + $ai;
                        $n++;
                    }
                }
            }
        }

        $this->out_act = $V2;

        return $this->out_act;
    }

    public function backward($y = null)
    {
        $V = $this->in_act; // we need to set dw of this
        $V2 = $this->out_act;
        $N = $this->out_depth;
        $V->dw = array_fill(0, count($V->w), 0); // zero out gradient wrt data

        // pass the gradient through the appropriate switch
        if ($this->out_sx === 1 && $this->out_sy === 1) {
            for ($i = 0; $i < $N; $i++) {
                $chain_grad = $V2->dw[$i];
                $V->dw[$this->switches[$i]] = $chain_grad;
            }
        } else {
            // bleh okay, lets do this the hard way
            $n = 0; // counter for switches
            for ($x = 0; $x < $V2->sx; $x++) {
                for ($y = 0; $y < $V2->sy; $y++) {
                    for ($i = 0; $i < $N; $i++) {
                        $chain_grad = $V2->get_grad($x, $y, $i);
                        $V->set_grad($x, $y, $this->switches[$n], $chain_grad);
                        $n++;
                    }
                }
            }
        }
    }
}

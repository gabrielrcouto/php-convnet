<?php
namespace ConvNet;

class PoolLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        // required
        $this->sx = $opt['sx']; // filter size
        $this->in_depth = $opt['in_depth'];
        $this->in_sx = $opt['in_sx'];
        $this->in_sy = $opt['in_sy'];

        // optional
        $this->sy = array_key_exists('sy', $opt) ? $opt['sy'] : $this->sx;
        $this->stride = array_key_exists('stride', $opt) ? $opt['stride'] : 2;
        $this->pad = array_key_exists('pad', $opt) ? $opt->pad : 0; // amount of 0 padding to add around borders of input volume

        // computed
        $this->out_depth = $this->in_depth;
        $this->out_sx = floor(($this->in_sx + $this->pad * 2 - $this->sx) / $this->stride + 1);
        $this->out_sy = floor(($this->in_sy + $this->pad * 2 - $this->sy) / $this->stride + 1);
        $this->layer_type = 'pool';
        // store switches for x,y coordinates for where the max comes from, for each output neuron
        $this->switchx = array_fill(0, $this->out_sx * $this->out_sy * $this->out_depth, 0);
        $this->switchy = array_fill(0, $this->out_sx * $this->out_sy * $this->out_depth, 0);
    }

    public function forward($V, $is_training)
    {
        $this->in_act = $V;

        $A = new Vol($this->out_sx, $this->out_sy, $this->out_depth, 0.0);

        $n = 0; // a counter for switches

        for ($d = 0; $d < $this->out_depth; $d++) {
            $x = -$this->pad;
            $y = -$this->pad;

            for ($ax = 0; $ax < $this->out_sx; $x += $this->stride, $ax++) {
                $y = -$this->pad;

                for ($ay = 0; $ay < $this->out_sy; $y += $this->stride, $ay++) {
                    // convolve centered at this particular location
                    $a = -99999; // hopefully small enough ;\
                    $winx = -1;
                    $winy = -1;

                    for ($fx = 0; $fx < $this->sx; $fx++) {
                        for ($fy = 0; $fy < $this->sy; $fy++) {
                            $oy = $y + $fy;
                            $ox = $x + $fx;

                            if ($oy >= 0 && $oy < $V->sy && $ox >= 0 && $ox < $V->sx) {
                                $v = $V->get($ox, $oy, $d);
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if ($v > $a) {
                                    $a = $v;
                                    $winx = $ox;
                                    $winy = $oy;
                                }
                            }
                        }
                    }

                    $this->switchx[$n] = $winx;
                    $this->switchy[$n] = $winy;
                    $n++;
                    $A->set($ax, $ay, $d, $a);
                }
            }
        }

        $this->out_act = $A;

        return $this->out_act;
    }

    public function backward($y = null)
    {
        // pooling layers have no parameters, so simply compute
        // gradient wrt data here
        $V = $this->in_act;
        $V->dw = array_fill(0, count($V->w), 0); // zero out gradient wrt data
        $A = $this->out_act; // computed in forward pass
        $n = 0;

        for ($d = 0; $d < $this->out_depth; $d++) {
            $x = -$this->pad;
            $y = -$this->pad;
            for ($ax = 0; $ax < $this->out_sx; $x += $this->stride, $ax++) {
                $y = -$this->pad;
                for ($ay = 0; $ay < $this->out_sy; $y += $this->stride, $ay++) {
                    $chain_grad = $this->out_act->get_grad($ax, $ay, $d);
                    $V->add_grad($this->switchx[$n], $this->switchy[$n], $d, $chain_grad);
                    $n++;
                }
            }
        }
    }
}

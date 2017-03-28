<?php
namespace ConvNet;

class ConvLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        // required
        $this->out_depth = $opt['filters'];
        // filter size. Should be odd if possible, it's cleaner.
        $this->sx = $opt['sx'];
        $this->in_depth = $opt['in_depth'];
        $this->in_sx = $opt['in_sx'];
        $this->in_sy = $opt['in_sy'];

        // optional
        $this->sy = array_key_exists('sy', $opt) ? $opt['sy'] : $this->sx;
        // stride at which we apply filters to input volume
        $this->stride = array_key_exists('stride', $opt) ? $opt['stride'] : 1;
        // amount of 0 padding to add around borders of input volume
        $this->pad = array_key_exists('pad', $opt) ? $opt['pad'] : 0;
        $this->l1_decay_mul = array_key_exists('l1_decay_mul', $opt) ? $opt['l1_decay_mul'] : 0.0;
        $this->l2_decay_mul = array_key_exists('l2_decay_mul', $opt) ? $opt['l2_decay_mul'] : 1.0;

        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        $this->out_sx = floor(($this->in_sx + $this->pad * 2 - $this->sx) / $this->stride + 1);
        $this->out_sy = floor(($this->in_sy + $this->pad * 2 - $this->sy) / $this->stride + 1);
        $this->layer_type = 'conv';

        // initializations
        $this->bias = array_key_exists('bias_pref', $opt) ? $opt['bias_pref'] : 0.0;
        $this->filters = [];

        for ($i = 0; $i < $this->out_depth; $i++) {
            $this->filters[] = new Vol($this->sx, $this->sy, $this->in_depth);
        }

        $this->biases = new Vol(1, 1, $this->out_depth, $this->bias);
    }

    public function forward($V, $is_training)
    {
        // optimized code by @mdda that achieves 2x speedup over previous version
        $this->in_act = $V;
        $A = new Vol($this->out_sx | 0, $this->out_sy | 0, $this->out_depth | 0, 0.0);

        $V_sx = $V->sx | 0;
        $V_sy = $V->sy |0;
        $xy_stride = $this->stride |0;

        for ($d = 0; $d < $this->out_depth; $d++) {
            $f = $this->filters[$d];
            $x = -$this->pad | 0;
            $y = -$this->pad | 0;

            for ($ay = 0; $ay < $this->out_sy; $y += $xy_stride, $ay++) {  // xy_stride
                $x = -$this->pad |0;

                for ($ax = 0; $ax < $this->out_sx; $x += $xy_stride, $ax++) {  // xy_stride
                    // convolve centered at this particular location
                    $a = 0.0;

                    for ($fy = 0; $fy < $f->sy; $fy++) {
                        $oy = $y + $fy; // coordinates in the original input array coordinates

                        for ($fx = 0; $fx<$f->sx; $fx++) {
                            $ox = $x + $fx;
                            if ($oy >= 0 && $oy < $V_sy && $ox >= 0 && $ox < $V_sx) {
                                for ($fd = 0; $fd < $f->depth; $fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    $a += $f->w[(($f->sx * $fy) + $fx) * $f->depth + $fd] * $V->w[(($V_sx * $oy) + $ox) * $V->depth + $fd];
                                }
                            }
                        }
                    }

                    $a += $this->biases->w[$d];
                    $A->set($ax, $ay, $d, $a);
                }
            }
        }

        $this->out_act = $A;

        return $this->out_act;
    }

    public function backward($y = null)
    {
        $V = $this->in_act;

        // zero out gradient wrt bottom data, we're about to fill it
        $V->dw = array_fill(0, count($V->w), 0);

        $V_sx = $V->sx | 0;
        $V_sy = $V->sy | 0;
        $xy_stride = $this->stride | 0;

        for ($d = 0; $d < $this->out_depth; $d++) {
            $f = $this->filters[$d];
            $x = -$this->pad | 0;
            $y = -$this->pad | 0;
            for ($ay = 0; $ay < $this->out_sy; $y += $xy_stride, $ay++) {  // xy_stride
                $x = -$this->pad | 0;
                for ($ax = 0; $ax < $this->out_sx; $x += $xy_stride, $ax++) {  // xy_stride
                    // convolve centered at this particular location
                    $chain_grad = $this->out_act->get_grad($ax, $ay, $d); // gradient from above, from chain rule
                    for ($fy = 0; $fy < $f->sy; $fy++) {
                        $oy = $y + $fy; // coordinates in the original input array coordinates
                        for ($fx = 0; $fx < $f->sx; $fx++) {
                            $ox = $x + $fx;
                            if ($oy >= 0 && $oy < $V_sy && $ox >= 0 && $ox < $V_sx) {
                                for ($fd = 0; $fd < $f->depth; $fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    $ix1 = (($V_sx * $oy) + $ox) * $V->depth + $fd;
                                    $ix2 = (($f->sx * $fy) + $fx) * $f->depth + $fd;
                                    $f->dw[$ix2] += $V->w[$ix1] * $chain_grad;
                                    $V->dw[$ix1] += $f->w[$ix2] * $chain_grad;
                                }
                            }
                        }
                    }
                    $this->biases->dw[$d] += $chain_grad;
                }
            }
        }
    }

    public function getParamsAndGrads()
    {
        $response = [];

        for ($i = 0; $i < $this->out_depth; $i++) {
            $response[] = [
                'params' => &$this->filters[$i]->w,
                'grads' => &$this->filters[$i]->dw,
                'l2_decay_mul' => &$this->l2_decay_mul,
                'l1_decay_mul' => &$this->l1_decay_mul
            ];
        }

        $response[] = [
            'params' => &$this->biases->w,
            'grads' => &$this->biases->dw,
            'l1_decay_mul' => 0.0,
            'l2_decay_mul' => 0.0
        ];

        return $response;
    }
}

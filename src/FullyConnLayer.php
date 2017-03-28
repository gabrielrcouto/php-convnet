<?php
namespace ConvNet;

class FullyConnLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        // required
        // ok fine we will allow 'filters' as the word as well
        $this->out_depth = array_key_exists('num_neurons', $opt) ? $opt['num_neurons'] : $opt['filters'];

        // optional
        $this->l1_decay_mul = array_key_exists('l1_decay_mul', $opt) ? $opt['l1_decay_mul'] : 0.0;
        $this->l2_decay_mul = array_key_exists('l2_decay_mul', $opt) ? $opt['l2_decay_mul'] : 1.0;

        // computed
        $this->num_inputs = $opt['in_sx'] * $opt['in_sy'] * $opt['in_depth'];
        $this->out_sx = 1;
        $this->out_sy = 1;
        $this->layer_type = 'fc';

        // initializations
        $this->bias = array_key_exists('bias_pref', $opt) ? $opt['bias_pref'] : 0.0;
        $this->filters = [];

        for ($i = 0; $i < $this->out_depth; $i++) {
            $this->filters[] = new Vol(1, 1, $this->num_inputs);
        }

        $this->biases = new Vol(1, 1, $this->out_depth, $this->bias);
    }

    public function forward($V, $is_training)
    {
        $this->in_act = $V;
        $A = new Vol(1, 1, $this->out_depth, 0.0);
        $Vw = $V->w;

        for ($i = 0; $i < $this->out_depth; $i++) {
            $a = 0.0;
            $wi = $this->filters[$i]->w;

            for ($d = 0; $d < $this->num_inputs; $d++) {
                $a += $Vw[$d] * $wi[$d]; // for efficiency use Vols directly for now
            }

            $a += $this->biases->w[$i];
            $A->w[$i] = $a;
        }

        $this->out_act = $A;

        return $this->out_act;
    }

    public function backward($y = null)
    {
        $V = $this->in_act;
        // zero out the gradient in input Vol
        $V->dw = array_fill(0, count($V->w), 0);

        // compute gradient wrt weights and data
        for ($i = 0; $i < $this->out_depth; $i++) {
            $tfi = $this->filters[$i];
            $chain_grad = $this->out_act->dw[$i];

            for ($d = 0; $d < $this->num_inputs; $d++) {
                $V->dw[$d] += $tfi->w[$d] * $chain_grad; // grad wrt input data
                $tfi->dw[$d] += $V->w[$d] * $chain_grad; // grad wrt params
            }

            $this->biases->dw[$i] += $chain_grad;
        }
    }

    public function getParamsAndGrads()
    {
        $response = [];

        for ($i = 0; $i < $this->out_depth; $i++) {
            $response[] = [
                'params' => &$this->filters[$i]->w,
                'grads' => &$this->filters[$i]->dw,
                'l1_decay_mul' => &$this->l1_decay_mul,
                'l2_decay_mul' => &$this->l2_decay_mul
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

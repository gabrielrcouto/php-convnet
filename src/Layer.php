<?php
namespace ConvNet;

class Layer
{
    public $out_depth;
    public $sx;
    public $in_depth;
    public $in_sx;
    public $in_sy;
    public $sy;
    public $stride;
    public $pad;
    public $l1_decay_mul;
    public $l2_decay_mul;
    public $out_sx;
    public $out_sy;
    public $layer_type;
    public $in_act;
    public $out_act;
    public $drop_prob;
    public $dropped;
    public $group_size;
    public $switches;
    public $filters;
    public $biases;
    public $bias;

    public function forward($V, $is_training)
    {
        return;
    }

    public function backward($y = null)
    {
        return;
    }

    public function fromJson($json)
    {
        foreach ($json as $key => $value) {
            if ($key === 'filters' && is_array($value)) {
                foreach ($value as $filter) {
                    if ($this->layer_type === 'conv') {
                        $vol = new Vol($json['sx'], $json['sy'], $json['in_depth']);
                    } else if ($this->layer_type === 'fc') {
                        $vol = new Vol(1, 1, $json['num_inputs']);
                    }

                    $vol->fromJson($filter);
                    $this->filters[] = $vol;
                }
            } else if ($key === 'biases') {
                $bias = array_key_exists('bias', $json) ? $json['bias'] : $json['biases'];

                $vol = new Vol(1, 1, $json['out_depth'], $bias);
                $vol->fromJson($value);
                $this->biases = $vol;
            } else {
                $this->$key = $value;
            }
        }
    }

    public function getParamsAndGrads()
    {
        return [];
    }

    public function setParamsAndGrads($params, $grads)
    {
        return;
    }
}

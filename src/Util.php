<?php
namespace ConvNet;

class Util
{
    public static function getopt($opt, $field_name, $default_value)
    {
        if (is_string($field_name)) {
            // case of single string
            return array_key_exists($field_name, $opt) ? $opt[$field_name] : $default_value;
        } else {
            // assume we are given a list of string instead
            $ret = $default_value;
            for ($i = 0; $i < count($field_name); $i++) {
                $f = $field_name[$i];
                if (array_key_exists($f, $opt)) {
                    // overwrite return value
                    $ret = $opt[$f];
                }
            }
        }

        return $ret;
    }
}

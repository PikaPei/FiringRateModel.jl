function f_tanh(x)
    return tanh(x)
end


function f_sigmoid(x)
    return 1 / (1 + exp(-x))
end


function f_sqrt(x)
    if x <= 0
        return 0
    else
        return sqrt(x)
    end
end


@register f_tanh(x)
@register f_sigmoid(x)
@register f_sqrt(x)


# NOTE: parameters?

from low import *


def linear_regression():
    # Build graph
    x = PlaceHolder(shape=(10, 1), name='x')
    y_true = PlaceHolder(shape=(1, 1), name='y_true')
    weight = Parameter(np.random.rand(1, 10), name='weight')
    bias = Parameter(np.random.rand(1, 1), name='bias')

    v = MatMul(weight, x, name='wx')
    y_pred = Add(v, bias, name='y_pred')

    # Loss
    minus_one = Constant(np.array([-1]), name='-1')
    minus_y = ScalarMul(minus_one, y_true, name='-y')
    diff = Add(y_pred, minus_y, name='y-y_true')
    loss = MatMul(diff, diff, name='loss')

    loss.show()

    # Train
    real_w = np.random.rand(1, 10)
    real_b = np.random.rand(1, 1)
    for i in range(10000):
        # generate and fill data
        real_x = np.random.rand(10, 1)
        real_y = real_w.dot(real_x) + real_b
        x.fill_value(real_x)
        y_true.fill_value(real_y)

        loss_val = loss.forward()[0][0]
        w_grad = weight.backward()
        b_grad = bias.backward()

        weight.simple_apply_grad(0.01)
        bias.simple_apply_grad(0.01)

        loss.reset_upstream()
        weight.grad.reset_upstream()
        bias.grad.reset_upstream()

        print(f"{i}   {loss_val}")
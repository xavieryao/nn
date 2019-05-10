from low import *
from medium import *


def xor():
    # define parameters
    x_in = PlaceHolder(shape=(2, 1), name='x')
    y_in = PlaceHolder(shape=(1, 1), name='y_true')
    w1 = Parameter(np.random.rand(32, 2), name='w1')
    b1 = Parameter(np.random.rand(32, 1), name='b1')
    w2 = Parameter(np.random.rand(1, 32), name='w2')
    b2 = Parameter(np.random.rand(1, 1), name='b2')
    params = [w1, b1, w2, b2]

    # forward calculation
    x = relu(w1 @ x_in + b1)
    x = sigmoid(w2 @ x + b2)

    # loss function
    loss = binary_cross_entropy(x, y_in)

    # print computation graph
    loss.show()

    # train
    iterations = 10000000
    for i in range(iterations):
        x1 = int(np.random.rand() > 0.5)
        x2 = int(np.random.rand() > 0.5)
        y_true = x1 ^ x2

        x_in.fill_value(np.asarray([x1, x2], dtype=np.float).reshape((2, 1)))
        y_in.fill_value(np.asarray([[y_true]], dtype=np.float))

        loss_val = loss.forward().item()
        prob = x.value.item()
        for p in params:
            p.backward()

        lr = 0.001
        for p in params:
            p.simple_apply_grad(lr)

        loss.reset_upstream()
        for p in params:
            p.grad.reset_upstream()

        print(f"It {i}/{iterations}   Loss: {loss_val}   y: {y_true}   logit: {prob}")


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
        weight.backward()
        bias.backward()

        weight.simple_apply_grad(0.01)
        bias.simple_apply_grad(0.01)

        loss.reset_upstream()
        weight.grad.reset_upstream()
        bias.grad.reset_upstream()

        print(f"{i}   {loss_val}")

if __name__ == '__main__':
    xor()
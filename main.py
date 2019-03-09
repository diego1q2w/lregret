import argparse

from datasets.linear.systolic_blood.lr import SystolicBloodProblem
from datasets.linear.parking.lr import ParkingProblem
from datasets.linear.national_unemployment.lr import UnemploymentProblem
from datasets.linear.fire_and_theft.lr import FireAndTheftProblem
from datasets.linear.computer_hadware.lr import ComputerHardwareProblem
from regresion.linear.feature import PolFeatures

from regresion.linear.linear import LinearRegression

problems = {
    'systolic_blood': SystolicBloodProblem,
    'parking': ParkingProblem,
    'unemployment': UnemploymentProblem,
    'fire_and_theft': FireAndTheftProblem,
    'computer_hardware': ComputerHardwareProblem,
}


def get_arguments() -> argparse.Namespace:
    p_names = list(problems.keys())

    parser = argparse.ArgumentParser("lRegret")
    parser.add_argument("--dataset", default=p_names[0],
                        choices=p_names,
                        help="The dataset to be used for training (default: %(default)s)")

    parser.add_argument("--operation", default='fit', type=str,
                        help="the kind of training to be executed on the dataset (default: %(default)s)")

    parser.add_argument("--lrate", default=0.000001, type=float,
                        help="learning rate for gradient descent (default: %(default)s)")

    parser.add_argument("--degree", default=1, type=int, help="Degree for polynomial regression")
    parser.add_argument("--l1", default=0.3, type=float, help="L1 regularisation constant")
    parser.add_argument("--l2", default=1000.0, type=float, help="L2 regularisation constant")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    lr = LinearRegression(learning_rate=args.lrate)
    problem = problems[args.dataset](lr)

    operation = getattr(problem, args.operation)

    if callable(operation):
        def arguments(x):
            return {
                'fit_l1': dict(l1=args.l1),
                'fit_l2': dict(l2=args.l2),
                'fit_polynomial': dict(pol_features=PolFeatures(args.degree)),
                'b': 2
            }.get(x, None)

        o_args = arguments(args.operation)
        if o_args:
            operation(o_args)
        else:
            operation()
    else:
        raise Exception("Not valid operation")
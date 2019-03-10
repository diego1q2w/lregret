import argparse

from typing import Dict

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


def call_operation(args: argparse.Namespace) -> bool:
    lr = LinearRegression(learning_rate=args.lrate)
    pf = PolFeatures(args.degree)
    problem = problems[args.dataset](lr, pol_features=pf)

    operation = getattr(problem, args.operation)

    def with_arguments(x) -> Dict:
        return {
            'fit_l1': dict(l1=args.l1),
            'fit_l2': dict(l2=args.l2),
        }.get(x, {})

    o_args = with_arguments(args.operation)

    if callable(operation):
        operation(**o_args)

    return callable(operation)


if __name__ == '__main__':
    u_args = get_arguments()

    if not call_operation(u_args):
        raise Exception("Invalid operation")

#!/usr/bin/env python3
from notears.notears import linear, utils
import numpy as np
import argparse


def main(args):
    X = np.loadtxt(args.X_path, delimiter=',')
    W_est = linear.notears_linear(X, lambda1=args.lambda1, loss_type=args.loss_type)
    assert utils.is_dag(W_est)
    np.savetxt(args.W_path, W_est, delimiter=',')


def parse_args():
    parser = argparse.ArgumentParser(description='Run NOTEARS algorithm')
    parser.add_argument('X_path', type=str, help='n by p data matrix in csv format')
    parser.add_argument('--lambda1', type=float, default=0.1, help='L1 regularization parameter')
    parser.add_argument('--loss_type', type=str, default='l2', help='l2, logistic, poisson loss')
    parser.add_argument('--W_path', type=str, default='W_est.csv', help='p by p weighted adjacency matrix of estimated DAG in csv format')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)


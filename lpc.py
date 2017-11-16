import sys
import math
import numpy


R = 6373.0

valid_stations = set([line.split('-')[0] for line in open('files.txt', 'r').readlines()])


def get_s_vector(data_map):
    col_vec = numpy.matrix(numpy.zeros((len(data_map), 1)))
    for i in data_map:
        filename = 'gz-data/%s-99999-2017.op' % i
        line = open(filename, 'r').readlines()[1]
        temperature = float(line.split(' ')[7])
        col_vec[data_map[i], 0] = temperature
    return col_vec


def read_csv_places(filename):
    temp_data = open('temps.txt', 'w')
    with open(filename) as f:
        lines = f.readlines()[1:]
    data = []
    data_map = {}
    c = 0
    for i in lines:
        points = i.replace('"', '').replace('\n', '').split(',')
        if points[3] != "IN":
            continue
        if points[6] == "" or points[7] == "":
            continue
        if points[0] not in valid_stations:
            continue
        data.append([int(points[0]), float(points[6]), float(points[7])])
        data_map[points[0]] = c
        c += 1
    data = numpy.matrix(data)
    data[:, 1] = data[:, 1] / 360 * 2 * numpy.pi
    temp_data.close()
    return data, data_map


def distance(p1, p2):
    lon1 = p1[0, 2]
    lon2 = p2[0, 2]
    dlon = lon2 - lon1
    lat1 = p1[0, 1]
    lat2 = p2[0, 1]
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_adjacency(data, k=3):
    adj_mat = numpy.asmatrix(numpy.zeros((data.shape[0], data.shape[0])))
    k += 1
    distances = []
    for idx in range(data.shape[0]):
        i = data[idx]

        # lat-lon distance
        lat_diff = data[:, 0] - i[0, 0]
        lon_diff = data[:, 1] - i[0, 1]
        res = numpy.square(numpy.sin(lat_diff / 2)) + numpy.multiply(
            numpy.multiply(numpy.square(numpy.sin(lon_diff / 2)), numpy.cos(data[:, 0])), numpy.cos(i[0, 0]))
        dist = numpy.arctan2(numpy.sqrt(res), numpy.sqrt(1 - res))

        distances.append(dist.T * R)

        # Euclidean distance
        # dist = numpy.sum(numpy.square(data - i), axis=1)

        # Perform k-NN
        top_idx = numpy.argpartition(dist.T, (1, k))[:, :k]

        for j in range(k):
            idy = top_idx[0, j]
            if idx == idy:
                continue
            adj_mat[idx, idy] = 1
            adj_mat[idy, idx] = 1
    distances = numpy.stack(distances)

    distances_adj = numpy.multiply(adj_mat, numpy.exp(-numpy.square(distances)))

    for idx in range(data.shape[0]):
        for idy in range(idx):
            if distances_adj[idx, idy] == 0:
                continue
            distances_adj[idx, idy] /= numpy.sqrt(numpy.sum(numpy.multiply(numpy.exp(-distances_adj[idx, :]), adj_mat[idx, :])))
            distances_adj[idx, idy] /= numpy.sqrt(numpy.sum(numpy.multiply(numpy.exp(-distances_adj[:, idy]), adj_mat[:, idy])))
            distances_adj[idy, idx] = distances_adj[idx, idy]

    return distances_adj


def get_b_matrix(A, s, l):
    res = []
    for i in range(1, l):
        A_pow = A ** i
        res.append(A_pow * s)
    return numpy.column_stack(res)


def get_h_mat(B, s):
    return numpy.linalg.pinv(B) * s


def apply_filter(A, h):
    res = numpy.asmatrix(numpy.zeros(A.shape))
    for i in range(0, h.shape[0]):
        res += A ** (i + 1) * h[i, 0]
    return res


def get_residual(A, h, s):
    h_A = apply_filter(A, h)
    return (numpy.eye(h_A.shape[0]) - h_A) * s


def get_signal_from_residual(A, h, r_):
    h_A = apply_filter(A, h)
    return numpy.linalg.inv((numpy.eye(h_A.shape[0]) - h_A)) * r_


def get_error(data, s, l, k):
    A = get_adjacency(data[:, 1:], k)
    b_mat = get_b_matrix(A, s, l)
    h_mat = get_h_mat(b_mat, s)
    r = get_residual(A, h_mat, s)
    signal = get_signal_from_residual(A, h_mat, r)
    return numpy.sum(numpy.square(signal - s)) / A.shape[0]


def main():
    places_filename = sys.argv[1]
    data, data_map = read_csv_places(places_filename)
    s = get_s_vector(data_map)

    opt_point = None
    for k in range(1, 16):
        for l in range(2, 11):
            error = get_error(data, s, l, k)
            if opt_point is None:
                opt_point = (l, k, error)
            elif opt_point[2] > error:
                opt_point = (l, k, error)
    print('Best k=%s and l=%s with error %s' % opt_point)


if __name__ == '__main__':
    main()

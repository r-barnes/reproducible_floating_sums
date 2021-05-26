import torch
import torch.cuda.profiler as profiler
torch.ops.load_library("build/libdeterministic.so")

# a = torch.zeros((4,5), dtype=torch.float).cuda()
# b = torch.tensor([
#   [1,2,3,4,5],
#   [5,4,3,2,1],
#   [9,8,7,6,5],
#   [0,1,1,2,3],
# ], dtype=torch.float).cuda()
# index = torch.tensor([0, 0, 2, 3]).cuda()

#a = torch.randint(0, 10, (3,3,3), dtype=torch.float).cuda()

# a = torch.ones((3,3,3), dtype=torch.float).cuda()
# b = torch.ones((3,3,3), dtype=torch.float).cuda()
# index = torch.tensor([2,0,2]).cuda()

dim = 1
shape = (20,30,400,500)
a = torch.randint(0, 10, shape, dtype=torch.float).cuda()
b = torch.randint(0, 10, shape, dtype=torch.float).cuda()
index = torch.randint(0, shape[dim], (shape[dim], ), dtype=torch.long).cuda()

current = a.clone().index_add_(dim, index, b)

# new = torch.ops.my_ops.index_add_deterministic(a, index, b)

profiler.start()
new = torch.ops.my_ops.index_add_deterministic_yuguo(a, dim, index, b)
profiler.stop()

# print("Correct\n", current)
# print("Mine\n", new)
print("Exact Equal?", torch.all(torch.eq(current, new)).cpu())
print("Close?", torch.allclose(current, new))

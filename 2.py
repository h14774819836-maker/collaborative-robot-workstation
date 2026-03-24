# 原始嵌套数组
nested_arrays = [[1, 2, 3]]

# 使用中间元素作为键进行降序排序
sorted_arrays = sorted(nested_arrays, key=lambda x: x[1], reverse=True)

# 输出排序后的结果
print("排序后的数组：")
for arr in sorted_arrays:
    print(arr)
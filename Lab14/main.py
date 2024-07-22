N = int(input())
numbers = [int(input()) for _ in range(N)]
control_value = int(input())

max_product = -1

for i in range(N):
    for j in range(i + 1, N):
        product = numbers[i] * numbers[j]
        if product % 21 == 0:
            max_product = max(max_product, product)

calculated_value = max_product if max_product != -1 else -1

print(f"Вычисленное контрольное значение: {calculated_value}")
if calculated_value == control_value:
    print("Контроль пройден")
else:
    print("Контроль не пройден")


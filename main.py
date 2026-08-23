
A = 32
B = 387

result = 0
total_token = 0
for i in range(8):
    total_token += A
    result += A * total_token
    if i >= 6:
        total_token += B * 3
        result += (B * 3) * total_token

print(result)

import math
print(math.sqrt(result))

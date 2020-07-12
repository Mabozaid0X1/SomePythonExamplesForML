# C96: C100
# C96--->https://www.youtube.com/watch?v=rh-Cwjma86w&list=PL6-3IRz2XF5UM-FWfQeF1_YhMMa12Eg3s&index=96
# C97--->https://www.youtube.com/watch?v=6ZtXy5LvpUE&list=PL6-3IRz2XF5UM-FWfQeF1_YhMMa12Eg3s&index=97
# C98--->https://www.youtube.com/watch?v=SWoG-Xxts2M&list=PL6-3IRz2XF5UM-FWfQeF1_YhMMa12Eg3s&index=98
# C99--->https://www.youtube.com/watch?v=T05CsB7R2Z0&list=PL6-3IRz2XF5UM-FWfQeF1_YhMMa12Eg3s&index=99
# C100-->https://www.youtube.com/watch?v=upxTVNw3hlU&list=PL6-3IRz2XF5UM-FWfQeF1_YhMMa12Eg3s&index=100

import math
import numpy as np
import random as rn
import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
# first example

mylist = [[1, 2, 3], [4, 5, 6]]
myarray = np.array(mylist)
print(("Matrix : %s") % str(myarray))  # array to string
print(("Matrix dimension is : %s") % str(myarray.shape))
print(("First row : %s") % str(myarray[0]))
print(("Last row : %s") % str(myarray[-1]))
print(("First row last column : %s") % str(myarray[0, 2]))
print(("Third column : %s") % str(myarray[:, 2]))

# second example
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "oct", "Nov", "Dec"]
sun = [44.7, 65.4, 101.7, 148.3, 170.9, 171.4,
       176.7, 186.1, 133.9, 105.4, 59.6, 45.8]
for s, m in sorted(zip(sun, months), reverse=True):
    print('{}: {:5.1f} hrs'.format(m, s))

# third example
FB = []
for t in range(210):
    if t % 5 == 0 and t % 3 == 0:
        FB.append("fizzbuzz")
    elif t % 5 == 0 and t % 3 != 0:
        FB.append("buzz")
    elif t % 5 != 0 and t % 3 == 0:
        FB.append("fizz")
    else:
        FB.append(t)
print(FB)

# fourth example
# pi = 0
# for k in range(20):
#     pi += pow(-3, -k) / (2*k+1)  # pow رقم أس رقم

# pi *= math.sqrt(12)
# print('pi = ', pi)

# print('error = ', abs(pi - math.pi))

# example 5

# x = []
# y = []


# def calcpi(n):
#     counter = 0
#     for k in range(1, n+2):
#         x.append(rn.random())
#         y.append(rn.random())

#     for gg in range(1, n+1):
#         for jj in range(1, n+1):
#             if ((x[gg])**2) + ((y[jj])**2) <= 1:
#                 counter += 1

#     pii = (float(counter)) / (n**2)
#     return pii


# calculatedpi = 4*calcpi(10000)
# print('calculated Pi = ' + str(calculatedpi))
# print('Differene = ' + str(np.pi - calculatedpi))


# example 6

xmin, xmax = -2. * math.pi, 2. * math.pi
n = 1000
x = [0.] * n
y = [0.] * n
dx = (xmax - xmin)/(n-1)
for i in range(n):
    xpt = xmin + i * dx
    x[i] = xpt
    y[i] = math.sin(xpt)**2

pylab.plot(x, y)
pylab.show()

# example 7

years = range(2000, 2010)
divorce_rate = [5.0, 4.7, 4.6, 4.4, 4.3,
                4.1, 4.2, 4.2, 4.2, 4.1]
margarine_consumption = [8.2, 7, 6.5, 5.3, 5.2,
                         4, 4.6, 4.5, 4.2, 3.7]

line1 = pylab.plot(years, divorce_rate, 'b-o',
                   label='Divorce rate in Maine')
pylab.ylabel('Divorces per 1000 people')
pylab.legend()

pylab.twinx()
line2 = pylab.plot(years, margarine_consumption, 'r-o',
                   label='Margarine cons')
pylab.ylabel('lb of Margarine (per capita)')

lines = line1 + line2
labels = []
for line in lines:
    labels.append(line.get_label())

pylab.legend(lines, labels)
pylab.show()


# example 8

body = {'Sun': (1.988e30, 5.955e5),
        'Mercury': (3.301e23, 2440.),
        'Venus': (4.867e+24, 6052.),
        'Earth': (5.972e24, 6371),
        'Mars': (6.417e23, 3390.),
        'Jupiter': (1.899e27, 69911.),
        'Saturn': (5.685e26, 58232.),
        'Uranus': (8.682e25, 25362.),
        'Neptune': (1.024e26, 24622.)}
planets = list(body.keys())
# The sun isn't a planet!
planets.remove('Sun')


def calc_density(m, r):
    # Returns the density of a sphere with mass m and radius r.
    return m / (4/3 * math.pi * r**3)


rho = {}
for planet in planets:
    m, r = body[planet]
    # calculate the density in g/cm3
    rho[planet] = calc_density(m*1000, r*1.e5)

for planet, density in sorted(rho.items()):
    print('The density of {0} is {1:3.2f} g/cm3'.format(planet, density))


# example 9

def makeT(mm):
    rr = len(mm)
    cc = len(mm[0])
    tm = [[0 for i in range(rr)] for j in range(cc)]  #
    for ccc in range(cc):
        for rrr in range(rr):
            tm[ccc][rrr] = mm[rrr][ccc]

    return tm


m = [[1, 20, 3, 4],
     [4, 5, 61, 8],
     [7, 8, 9, 60],
     [30, 3, 6, 3],
     [0, 7, 60, 9]]

print(makeT(m))
# يوجد هذا الأمر في numpy (.T) Transpose مقلوب المصفوفة
print(np.transpose(m))


# example 10


def wtm(t):
    txt = ''
    for i in t.upper():
        try:
            txt += morse.get(i)

        except:
            pass

    return txt


morse = dict((('A', '.-'), ('B', '-...'), ('C', '-.-.'), ('D', '--.'),
              ('E', '.'), ('F', '..-.'), ('G', '--.'), ('H', '....'),
              ('I', '..'), ('J', '.---'), ('K', '-.-'), ('L', '.-..'),
              ('M', '--'), ('N', '-.'), ('O', '---'), ('P', '.--.'),
              ('Q', '--.-'), ('R', '.-.'), ('S', '...'), ('T', '-'),
              ('U', '..-'), ('V', '...-'), ('W', '.--'), ('X', '-..-'),
              ('Y', '-.--'), ('Z', '--..'), ('1', '.----'), ('2', '..---'),
              ('3', '...--'), ('4', '....-'), ('5', '.....'), ('6', '-....'),
              ('7', '--...'), ('8', '---..'), ('9', '----.'), ('0', '-----'),
              (' ', '/')))

print(wtm('Hello world'))

# example 11


def lreg(xx, yy):
    xdash = pylab.mean(xx)
    ydash = pylab.mean(yy)
    z = []
    for g in range(len(xx)):
        z.append(float(xx[g]) * float(yy[g]))
    w = []
    for gg in range(len(xx)):
        w.append(float(xx[gg]) * float(xx[gg]))
    xydash = pylab.mean(z)
    x2dash = pylab.mean(w)
    m = (xydash - (xdash * ydash)) / (x2dash - (xdash**2))
    c = ydash - (m*xdash)
    return round(m, 5), round(c, 5)


xdata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ydata = [4.8, 5, 7.5, 7.5, 9.4, 9.5, 11.7, 12, 11.2, 15]
mm, cc = lreg(xdata, ydata)
print('Slope is (' + str(mm) + ')')
print('C Value is (' + str(cc) + ')')
exactx = []
exacty = []
for i in range(12):
    exactx.append(i)
    exacty.append((cc + (i * mm)))
pylab.plot(xdata, ydata, 'o', markersize=6, color='b')
pylab.plot(exactx, exacty, linewidth=2, color='r',
           linestyle='-', label='predicted')
pylab.show()


# example 12

Polynomial = np.polynomial.Polynomial
conc = np.array([0, 20, 40, 80, 120, 180, 260, 400, 800, 1500])
A = np.array([2.287, 3.528, 4.336, 6.909, 8.274,
              12.855, 16.085, 24.797, 49.058, 89.400])

cmin, cmax = min(conc), max(conc)
pfit, stats = Polynomial.fit(
    conc, A, 1, full=True, window=(cmin, cmax), domain=(cmin, cmax))

print('Raw fit results:', pfit, stats)

A0, m = pfit
resid, rank, sing_val, rcond = stats
rms = np.sqrt(resid[0]/len(A))

print('Fit: A = {:.3f}[P] + {:.3f}'.format(m, A0),
      '(rms residual = {:.4f})'.format(rms))

pylab.plot(conc, A, 'o', color='r')
pylab.plot(conc, pfit(conc), color='b')
pylab.xlabel('[P] /$\mathrm{\mu g\cdot mL^{-1}}$')
pylab.ylabel('Absorbance')
pylab.show()


# example 13

def gen_primes(N):
    primes = set()
    for n in range(2, N):
        if all(n % p > 0 for p in primes):  # all (like) and
            primes.add(n)
            yield n  # بيروح علي الفور بس بيأخد الي بعدها


print(*gen_primes(100))
# طريقة أخري


def pn(n):
    pnn = []
    for h in range(2, n+1):
        divisible = True
        for j in range(h-1, 1, -1):
            if (h % j) == 0:
                divisible = False
        if divisible:
            pnn.append(h)

    print(pnn)
    return pnn


pn(100)


# example 14

L, n = 2, 400
x = np.linspace(-L, L, n)
y = x.copy()
X, Y = np.meshgrid(x, y)  # لعمل شبكة مربعة من القيم
Z = np.exp(-(X**2 + Y**2))

fig, ax = plt.subplots(nrows=2, ncols=2,
                       subplot_kw={'projection': '3d'})
ax[0, 0].plot_wireframe(X, Y, Z, rstride=40, cstride=40)
ax[0, 1].plot_surface(X, Y, Z, rstride=40, cstride=40, color='m')
ax[1, 0].plot_surface(X, Y, Z, rstride=12, cstride=12, color='m')
ax[1, 1].plot_surface(X, Y, Z, rstride=20, cstride=20, cmap=cm.hot)
for axes in ax.flatten():
    axes.set_xticks([-2, -1, 0, 1, 2])
    axes.set_yticks([-2, -1, 0, 1, 2])
    axes.set_zticks([0, 0.5, 1])

fig.tight_layout()
plt.show()


# example 15

plt.style.use('ggplot')
countries = ['Brazil', 'Madagascar', 'S.Korea', 'United States',
             'Ethiopia', 'Pakistan', 'Chine', 'Belize']
# Birth rate per 1000 population
birth_rate = [16.4, 33.5, 9.5, 14.2, 38.6, 30.2, 13.5, 23.0]
# Life expectancy at birth years
life_expectancy = [73.7, 64.3, 81.3, 78.8, 63.0, 66.4, 75.2, 73.7]
# Per person income fixed to US Dollars in 2000
GDP = np.array([4800, 240, 16700, 37700, 230, 670, 2640, 3490])

fig = plt.figure()
ax = fig.add_subplot(111)

# Some random colors:
colors = range(len(countries))
ax.scatter(birth_rate, life_expectancy, c=colors, s=GDP*.1)  # s=size

ax.set_xlabel('Birth rate per 1000 population')
ax.set_ylabel('Life expectancy at birth (years)')
plt.show()

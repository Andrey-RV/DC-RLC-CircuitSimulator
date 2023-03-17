import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import Sequence

number = int | float
numeric_sequence = Sequence[int | float]


class DcNoSource:
    def __init__(self, R: number, L: number, C: number) -> None:
        self.R = R
        self.L = L
        self.C = C
        self.omega = 1 / np.sqrt(L * C)  # type: ignore


class SeriesRLC(DcNoSource):
    def __init__(self, R: number, L: number, C: number, i0: number, vl0: number, vc0: number, vr0: number) -> None:
        r"""
        Args:
            R (number): _The equivalent resistance of the circuit._
            L (number): _The equivalent inductance of the circuit._
            C (number): _The equivalent capacitance of the circuit._
            i0 (number): _The initial current common to all elements._
            vl0 (number): _The initial voltage across the inductor._
            vc0 (number): _The initial voltage across the capacitor._
            vr0 (number): _The initial voltage across the resistor._
        """
        super().__init__(R, L, C)
        self.i0 = i0
        self.vl0 = vl0
        self.vc0 = vc0
        self.vr0 = vr0
        self.i_prime0 = None
        self.alpha = R / (2 * L)


class ParallelRLC(DcNoSource):
    def __init__(self, R: number, L: number, C: number, v0: number, il0: number, ic0: number, ir0: number) -> None:
        r"""
        Args:
            R (int or float): _The equivalent resistance of the circuit._
            L (int or float): _The equivalent inductance of the circuit._
            C (int or float): _The equivalent capacitance of the circuit._
            v0 (int or float): _The initial voltage common to all elements._
            il0 (int or float): _The initial current through the inductor._
            ic0 (int or float): _The initial current through the capacitor._
            ir0 (int or float): _The initial current through the resistor._
        """
        super().__init__(R, L, C)
        self.v0 = v0
        self.il0 = il0
        self.ic0 = ic0
        self.ir0 = ir0
        self.v_prime0 = -(v0 + R * il0) / (R * C)
        self.alpha = 1 / (2 * R * C)
        self.__get_voltages()
        self.__get_capacitor_current()
        self.__get_inductor_current()
        self.__get_resistor_current()

    def __get_voltages(self) -> None:
        r"""_Get the expression for the voltage common to all elements._"""
        t = sp.Symbol('t')
        c1 = sp.Symbol('c1')
        c2 = sp.Symbol('c2')

        unsolved_voltage_expr = self.__get_voltage_expression(t, c1, c2)  # type: ignore
        unsolved_voltage_expr_prime = sp.diff(unsolved_voltage_expr, t)                           # type: ignore

        eq1 = sp.Eq(unsolved_voltage_expr.subs(t, 0), self.v0, evaluate=False)                                    # type: ignore
        eq2 = sp.Eq(unsolved_voltage_expr_prime.subs(t, 0), self.v_prime0, evaluate=False)                        # type: ignore
        solution = sp.solve([eq1, eq2], [c1, c2])                                                 # type: ignore

        self.voltage = unsolved_voltage_expr.subs([(c1, solution[c1]), (c2, solution[c2])])       # type: ignore

    def __get_voltage_expression(self, t: sp.Symbol, c1: sp.Symbol, c2: sp.Symbol) -> sp.Expr:
        """_Returns the unsolved expression (unknown constants) for the voltage common to all elements._

        Args:
            t (sp.Symbol): _A symbol representing the time._
            c1 (sp.Symbol): _A symbol representing the first constant._
            c2 (sp.Symbol): _A symbol representing the second constant._

        Returns:
            sp.Expr: _The unsolved expression for the voltage._
        """
        delta = self.alpha ** 2 - self.omega ** 2

        if delta > 0:
            voltage = self.__overdamped_response(t, c1, c2, delta)
        elif delta == 0:
            voltage = self.__critically_damped_response(t, c1, c2)
        else:
            voltage = self.__underdamped_response(t, c1, c2, delta)

        return voltage

    def __overdamped_response(self, t: sp.Symbol, c1: sp.Symbol, c2: sp.Symbol, delta: number) -> sp.Expr:
        r"""_Returns the expression for the voltage (with the unknown constants) for the case in which \(\alpha^2 > \omega^2\)_

        Args:
            t (sp.Expr): _A symbol representing the time._
            a1 (sp.Expr): _A symbol representing the first constant._
            a2 (sp.Expr): _A symbol representing the second constant._
            delta (int or float): The result of _\(\alpha^2 - \omega^2\)._

        Returns:
            sp.Expr: _The expression for the voltage for \(t>0\) with the unknown constants._
        """
        s1 = -self.alpha + sp.sqrt(delta)                    # type: ignore
        s2 = -self.alpha - sp.sqrt(delta)                    # type: ignore
        voltage = c1 * sp.exp(s1 * t) + c2 * sp.exp(s2 * t)  # type: ignore
        return sp.simplify(voltage)                          # type: ignore

    def __critically_damped_response(self, t: sp.Symbol, c1: sp.Symbol, c2: sp.Symbol) -> sp.Expr:
        r"""_Returns the expression for the voltage (with the unknown constants) for the case in which \(\alpha^2 = \omega^2\)._

        Args:
            t (sp.Symbol): _A symbol representing the time._
            c1 (sp.Symbol): _A symbol representing the first constant._
            c2 (sp.Symbol): _A symbol representing the second constant._

        Returns:
            sp.Expr: _The expression for the voltage for \(t>0\) with the unknown constants._
        """
        s = -self.alpha
        voltage = c1 * sp.exp(s * t) + c2 * t * sp.exp(s * t)  # type: ignore
        return sp.simplify(voltage)                            # type: ignore

    def __underdamped_response(self, t: sp.Expr, c1: sp.Expr, c2: sp.Expr, delta: number) -> sp.Expr:
        r"""_Returns the expression for the voltage (with the unknown constants) for the case in which \(\alpha^2 < \omega^2\)._

        Args:
            t (sp.Symbol): _A symbol representing the time._
            c1 (sp.Symbol): _A symbol representing the first constant._
            c2 (sp.Symbol): _A symbol representing the second constant._
            delta (int or float): The result of _\(\alpha^2 - \omega^2\)._

        Returns:
            sp.Expr: _The expression for the voltage for \(t>0\) with the unknown constants._
        """
        s = -self.alpha
        voltage = sp.exp(s * t) * (c1 * sp.cos(sp.sqrt(-delta) * t) + c2 * sp.sin(sp.sqrt(-delta) * t))  # type: ignore
        return sp.simplify(voltage)                                                                      # type: ignore

    def plot(self, quantity: str, time: numeric_sequence) -> None:
        r"""Plot the voltage or currents of the circuit for a given time interval.

        Args:
            quantity (str): _The quantity to be plotted. It can be either 'voltage' or 'current'._
            t (Sequence[int or float]): _The time interval for which the quantity is to be plotted._
        """
        fig, ax = plt.subplots()                  # type: ignore
        ax.spines['left'].set_position('zero')    # type: ignore
        ax.spines['bottom'].set_position('zero')  # type: ignore
        ax.spines['left'].set_linestyle('--')     # type: ignore
        ax.spines['bottom'].set_linestyle('--')   # type: ignore
        ax.spines['top'].set_visible(False)       # type: ignore
        ax.spines['right'].set_visible(False)     # type: ignore
        ax.spines['left'].set_color('black')      # type: ignore
        ax.spines['bottom'].set_color('black')    # type: ignore

        if quantity == 'voltage':
            voltages = [float(self.__v(t)) for t in time]
            ax.set_ylabel('V(V)', horizontalalignment='right', rotation='horizontal', labelpad=-10)  # type: ignore
            plt.plot(time, voltages, color='green')                                         # type: ignore

        elif quantity == 'current':
            capacitor_current = np.array([float(self.__i_c(t)) for t in time])
            inductor_current = np.array([float(self.__i_l(t)) for t in time])
            resistor_current = np.array([float(self.__i_r(t)) for t in time])
            ax.set_ylabel('I(A)', horizontalalignment='right', rotation='horizontal', labelpad=-10)  # type: ignore
            plt.plot(time[time < 0], capacitor_current[time < 0], color='red', label=r'$I_{C}(t)$')  # type: ignore
            plt.plot(time[time > 0], capacitor_current[time > 0], color='red')  # type: ignore
            plt.plot(time[time < 0], inductor_current[time < 0], color='blue', label=r'$I_{L}(t)$')   # type: ignore
            plt.plot(time[time > 0], inductor_current[time > 0], color='blue')  # type: ignore
            plt.plot(time[time < 0], resistor_current[time < 0], color='green', label=r'$I_{R}(t)$')  # type: ignore
            plt.plot(time[time > 0], resistor_current[time > 0], color='green')  # type: ignore

        ax.set_xlabel('t(s)', horizontalalignment='right', labelpad=-10)                             # type: ignore
        ax.xaxis.set_label_coords(1.04, 0.08)                                                        # type: ignore
        ax.yaxis.set_label_coords(0.16, 1.025)                                                       # type: ignore
        plt.legend(fontsize='large')                                                                 # type: ignore
        plt.show()                                                                                   # type: ignore

    def __get_capacitor_current(self) -> None:
        r"""_Get the expression for the capacitor current for \(t>0\)._"""
        t = sp.Symbol('t')
        self.capacitor_current = self.C * sp.diff(self.voltage, t)                                  # type: ignore

    def __get_inductor_current(self) -> None:
        r"""_Get the expression for the inductor current for \(t>0\)._"""
        t = sp.Symbol('t')
        self.inductor_current = (1 / self.L) * sp.integrate(self.voltage, (t, 0, t)) + self.il0     # type: ignore

    def __get_resistor_current(self) -> None:
        r"""_Get the expression for the resistor current for \(t>0\)._"""
        self.resistor_current = self.voltage / self.R  # type: ignore

    def __v(self, t: number) -> number:
        r"""_Returns v0 if \(t<0\) and self.voltage.subs(sp.Symbol('t'), t) if \(t\geq0\)._

        Args:
            t (int or float): _The time at which the voltage is to be calculated._

        Returns:
            int or float: _The voltage at time /(t/)._
        """
        if t < 0:
            return self.v0
        else:
            return self.voltage.subs(sp.Symbol('t'), t)  # type: ignore

    def __i_c(self, t: number) -> number:
        r"""_Returns \(I_{C}(0)\) if \(t<0\) and self.capacitor_current.subs(sp.Symbol('t'), t) if \(t\geq0\)._

        Args:
            t (int or float): _The time at which the current is to be calculated._

        Returns:
            int or float: _The current at time /(t/)._
        """
        if t < 0:
            return self.ic0
        else:
            return self.capacitor_current.subs(sp.Symbol('t'), t)  # type: ignore

    def __i_l(self, t: number) -> number:
        r"""_Returns \(I_{L}(0)\) if \(t<0\) and self.inductor_current.subs(sp.Symbol('t'), t) if \(t\geq0\)._

        Args:
            t (int or float): _The time at which the current is to be calculated._

        Returns:
            int or float: _The current at time /(t/)._
        """
        if t < 0:
            return self.il0
        else:
            return self.inductor_current.subs(sp.Symbol('t'), t)  # type: ignore

    def __i_r(self, t: number) -> number:
        r"""_Returns \(I_{R}(0)\) if \(t<0\) and self.resistor_current.subs(sp.Symbol('t'), t) if \(t\geq0\)._

        Args:
            t (int or float): _The time at which the current is to be calculated._

        Returns:
            int or float: _The current at time /(t/)._
        """
        if t < 0:
            return self.ir0
        else:
            return self.resistor_current.subs(sp.Symbol('t'), t)  # type: ignore

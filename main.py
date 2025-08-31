import logging
import time
from typing import NamedTuple

import pyperclip
from draftsman.blueprintable import Blueprint
from draftsman.classes.entity import Entity
from draftsman.classes.vector import Vector
from draftsman.constants import Direction
from factoriocalc import Box, config, itm, presets, produce, rcp, setGameConfig
from factoriocalc.fracs import Frac, div, frac
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

logger = logging.getLogger("main")


def sign(i):
    return 1 if i > 0 else -1 if i < 0 else 0


BELT_LIMIT = {
    "transport-belt": 15,
    "fast-transport-belt": 30,
    "express-transport-belt": 45,
}
INSERTER_LIMIT = [
    # TODO: actually determine these for various stack upgrades
    # normal, fast
    (frac("1.7"), 7),
]
INSERTER_LEVEL = 0
BELT_COLOURS = {
    "transport-belt": "yellow",
    "fast-transport-belt": "red",
    "express-transport-belt": "blue",
}
INSERTER_COLOURS = {
    "inserter": "yellow",
    "long-hand-inserter": "red",
    "fast-inserter": "blue",
    "bulk-inserter": "green",
}


def belt_required(rate):
    for belt, limit in BELT_LIMIT.items():
        if rate <= limit:
            return belt
    assert False, "rate exceeds single belt"


def belts_required(rate):
    for belt, limit in BELT_LIMIT.items():
        if rate <= limit:
            return (1, belt)
    n = round(rate / 45)  # round up
    return (n, "express-transport-belt")


def inserter_required(rate):
    normal, fast = INSERTER_LIMIT[INSERTER_LEVEL]
    if rate < normal:
        return "inserter"
    if rate < fast:
        return "fast-inserter"
    return "bulk-inserter"


def inserter_colour(i: str, text: str):
    return f"[{INSERTER_COLOURS[i]}]{text}[/]"


def belt_colour(rate):
    belt = belt_required(rate)
    return f"[{BELT_COLOURS[belt]}]{rate:.2f}/s[/]"


class Route:
    def __init__(self, start: Vector, belt: str):
        self.points = [start]
        self.belt = belt

    @property
    def start(self):
        return self.points[0]

    @property
    def end(self):
        return self.points[-1]

    def route_to(self, p):
        self.points.append(p)
        return self

    def route(self, x, y):
        if x and y:
            raise ValueError("Diagonals unsupported")
        return self.route_to(self.end + (x, y))


class Placer:
    def __init__(self, blueprint: Blueprint, origin: Vector, routes: list):
        self.blueprint = blueprint
        self.origin = origin
        self.routes = routes

    @property
    def x(self):
        return self.origin.x

    @property
    def y(self):
        return self.origin.y

    def entity(self, *args, **kwargs):
        rel_pos = (0, 0)
        name = None
        for arg in args:
            if isinstance(arg, str):
                name = arg
            elif isinstance(arg, tuple):
                rel_pos = arg
            elif isinstance(arg, Direction):
                kwargs["direction"] = arg

        return self.blueprint.entities.append(name=name, tile_position=self.origin + rel_pos, **kwargs)

    def place(self, name, *args, **kwargs):
        self.entity(name, *args, **kwargs)
        return self

    def inserter(self, *args, **kwargs):
        return self.place("inserter", *args, **kwargs)

    def long_inserter(self, *args, **kwargs):
        return self.place("long-handed-inserter", *args, **kwargs)

    # def belt(self, name: str, *args, **kwargs):
    #     return self.place(name: str, *args, **kwargs)

    def electric_pole(self, *args, **kwargs):
        return self.place("small-electric-pole", *args, **kwargs)

    def new_route(self, rate):
        belt = belt_required(rate)
        route = Route(self.origin, belt)
        self.routes.append(route)
        return route

    # def route_to(self, b, belt="transport-belt"):
    #     return self.route(b.x - self.x, b.y - self.y, belt)

    # def route(self, x, y, belt="transport-belt"):
    #     if x and y:
    #         raise ValueError("Diagonals unsupported")
    #     # All routing is performed at end together as complex.
    #     self.routes.append((self.origin, x, y, belt))
    #     return self[x, y]

    def render_routes(self):
        segments = []
        for route in self.routes:
            for a, b in zip(route.points, route.points[1:]):
                segments.append((a, b, route.belt))
        # render horizontals first, as undergrounding verticals over bus works better generally.
        segments.sort(key=lambda segment: segment[0].x - segment[1].x == 0)
        for segment in segments:
            self.render_segment(*segment)

    def render_segment(self, a, b, belt: str):
        ug = belt.replace("transport", "underground")
        x = b.x - a.x
        y = b.y - a.y
        d = S if y > 0 else N if y < 0 else E if x > 0 else W
        dx = sign(x)
        dy = sign(y)
        p = self.abs(a.x, a.y)
        end = p[x, y]
        underground = False
        for i in range(abs(x + y)):
            np = p[dx, dy]
            if underground:
                underground -= 1
                if underground == 0:
                    print("Underground length exhausted: %s" % p.origin)
                if not p.entity_at_position():
                    p.place(ug, d, io_type="output")
                    underground = False
            elif np.entity_at_position() and np.origin != end.origin:
                underground = 5
                p.place(ug, d)
            else:
                p.place(belt, d)
            p = np
        return p

    def underground(self, *args, **kwargs):
        return self.place("underground-belt", *args, **kwargs)

    def linked(self, *args, **kwargs):
        return self.place("linked-belt", *args, **kwargs)

    def splitter(self, belt, *args, **kwargs):
        name = belt.replace("transport-belt", "splitter")
        return self.place(name, *args, **kwargs)

    def entity_at_position(self):
        return self.blueprint.find_entity_at_position(self.origin + (0.5, 0.5))

    def __getitem__(self, p):
        return Placer(self.blueprint, self.origin + Vector(*p), self.routes)

    def __add__(self, rel_pos):
        return Placer(self.blueprint, self.origin + rel_pos, self.routes)

    def abs(self, x, y):
        return Placer(self.blueprint, Vector(x, y), self.routes)


W = Direction.WEST
E = Direction.EAST
N = Direction.NORTH
S = Direction.SOUTH


class BusLane(NamedTuple):
    item: Entity
    rate: Frac
    route: Route

    @property
    def odd(self):
        # Ugly (should lanes know which number they are?)
        return self.route.end.y % 2 == 0


FREE_LANE = BusLane(None, 0, None)


def lane_y(i: int):
    # ||  ||  ||
    # 0,1,4,5,8,9,12,13,...
    return i * 2 - i % 2


class LayoutEngine:
    vpad = 3

    def run(self, factory: Box, blueprint: Blueprint):
        logger.info("Running layout...")
        start = time.time()
        ordered = self.order(factory)
        origin = Placer(blueprint, Vector(0, 0), [])
        bus = []
        for i, item in enumerate(factory.inputs):
            # Calculate lane(s) required
            rate = factory.flow(item).rateIn
            n, belt = belts_required(rate)
            assert n == 1, "Multiple input belts not yet supported"
            route = origin[-3, lane_y(i)].new_route(rate)
            lane = BusLane(item, rate, route)

            logger.info(
                "↠ belt %d: %s %s",
                i,
                item,
                belt_colour(rate),
            )
            bus.append(lane)

        for mul in ordered:
            column_width = self.layout_column(origin, bus, mul)
            origin += (column_width, 0)

        # Carry out final bus lanes
        for lane in bus:
            if lane == FREE_LANE:
                continue
            lane.route.route(origin.x - lane.route.start.x, 0)

        # Finally actually render belts
        origin.render_routes()

        # Find bottom power poles and wire together
        poles = origin.blueprint.find_entities_filtered(name="small-electric-pole")
        lowest_y = max(p.position.y for p in poles)
        poles = [p for p in poles if p.position.y == lowest_y]
        pole_iter = iter(poles[1:])
        for a, b in zip(pole_iter, pole_iter):
            origin.blueprint.add_power_connection(a, b)

        took = (time.time() - start) * 100
        logger.info(f"✅ [green]Success[/] (took {took:.1f}ms)")

    def order(self, factory):
        group = factory.inner
        satisfied = set(factory.inputs)
        pending = group
        ordered = []

        while pending:
            # Find next machine ready
            ready = [m for m in pending if set(m.machine.inputs).issubset(satisfied)]
            assert ready

            first = ready[0]  # Potential for smarter selection here
            ordered.append(first)
            satisfied.update(first.outputs)
            pending = [m for m in pending if m is not first]

        return ordered

    def layout_column(self, origin, bus, mul):
        input_belts = len(mul.inputs)  # TODO: liquids
        output_belts = len(mul.outputs)  # TODO: chemical plants

        output_item = list(mul.outputs)[0]
        logger.info("[b]Column: %dx %s[/b] (%d%%)", mul.num, mul.recipe.name, mul.machine.throttle * 100)

        output_rate = mul.flow(output_item).rateOut
        output_per_machine = div(output_rate, mul.num)
        n, output_belt = belts_required(output_rate)
        assert n == 1, "Multiple output belts not supported"
        output_inserter = inserter_required(output_per_machine)

        left_belts = 1 if input_belts < 2 else 2
        right_belts = output_belts if input_belts < 3 else input_belts + output_belts - 2

        column_width = (
            left_belts + 1 + mul.machine.width + 1 + (3 if right_belts > 1 else 1)
        )  # Add extra space for the 3rd belt tap
        left = origin[left_belts + 1, -self.vpad + 1]

        start_entity_index = len(origin.blueprint.entities)

        input_items = list(mul.inputs)
        input_inserters = [inserter_required(div(mul.flow(item).rateIn, mul.num)) for item in input_items]

        # Machines, etc
        half = mul.num // 2
        for j in range(mul.num):
            if mul.num > 1 and j == half:
                # Make space for right-left balancer.
                left = left[0, -3]
                left[1, 0].electric_pole()

            left = left[0, -mul.machine.height]
            left.place(
                mul.machine.name,
                N,
                recipe=mul.machine.recipe.name,
            )

            # Input inserter(s)
            if input_belts > 1:
                left[-1, 1].place(input_inserters[1], W)
                left[-1, 2].long_inserter(W)
            else:
                left[-1, 1].place(input_inserters[0], W)

            right = left[mul.machine.width, 0]
            if input_belts > 2:
                # Extra input inserter to pull from rightmost input belt
                right[0, 2].long_inserter(E)

            # Output inserter
            right[0, 1].place(output_inserter, W)

            # Power poles, every other
            if j % 2 == 0:
                left[-1, 0].electric_pole()
                right[0, 0].electric_pole()

        left = left[0, -mul.machine.height]
        column_height = -(left.origin.y + 2)

        # Inputs
        right = left_belts + 1 + mul.machine.width + 1

        active_lanes = {i for i, lane in enumerate(bus) if lane.item in mul.inputs}

        for i, input_item in enumerate(input_items):
            # Find lane
            j, lane = next((i, lane) for i, lane in enumerate(bus) if lane.item == input_item)

            # Calculate consumption from lane
            consumed = mul.flow(input_item).rateIn
            remain = lane.rate - consumed
            fully_consumed = remain == 0

            inserter_type = input_inserters[i]
            per_machine = div(consumed, mul.num)
            logger.info(
                "↠ %s total: %s, per machine: %s",
                input_item,
                belt_colour(consumed),
                inserter_colour(inserter_type, "%.2f/s" % (per_machine,)),
            )

            x = i if i < left_belts else right + i - 1
            s = origin[x, lane.route.end.y]
            end = origin[x, -column_height + 1]
            if fully_consumed:
                # Fully consumed - terminate bus lane
                # logger.info("🫙 Fully consumed: '%s'" % input_item)
                if lane.odd:
                    # Take bus lane straight into input
                    lane.route.route_to(s.origin).route_to(end.origin)
                else:
                    # 'Pig tail' loop round and under
                    # -+\
                    #  ^|
                    #  \/
                    lane.route.route_to(s[1, 0].origin).route(0, 2).route(-1, 0).route_to(end.origin)
                # Free up slot
                bus[j] = FREE_LANE
                continue

            # Bus tap
            if lane.odd:
                #   >/
                #  ->+
                s.splitter(lane.route.belt, (-1, -1), E)
                # Route lane into splitter
                lane.route.route_to(s[-1, 0].origin)
                # And top output of splitter to input
                top_output = s[0, -1].new_route(consumed)
                top_output.route_to(end.origin)
            else:
                #  +>-
                #  ^>\
                #  \-/
                # Special case - if the next odd lane is being tapped too, it'd clash.
                if j + 1 in active_lanes:
                    # .>/,
                    #  >+
                    s.splitter(lane.route.belt, (-1, -1), E)
                    # Route lane into splitter
                    lane.route.route_to(s[-1, 0].origin)
                    # And top output of splitter to input
                    s[0, -1].place(belt_required(consumed), N)  # yuck
                    top_output = s[0, -2].new_route(consumed)
                    top_output.route_to(end.origin)
                else:
                    s.splitter(lane.route.belt, (1, 0), E)
                    # Route lane into splitter
                    lane.route.route_to(s[1, 0].origin)
                    # And top output of splitter to input
                    top_output = s[2, 1].new_route(consumed)
                    top_output.route(0, 1).route(-2, 0).route_to(end.origin)
                    # Move continuation point to splitter out
                    s = s[2, 0]

            # Bottom output of splitter
            new_bus_route = s.new_route(remain)
            # Update remaining and start
            bus[j] = lane._replace(rate=remain, route=new_bus_route)

        logger.info(
            "↞ %s: total %s, per machine: %s",
            output_item,
            belt_colour(output_rate),
            inserter_colour(output_inserter, "%.2f/s" % (output_per_machine,)),
        )
        # Output(s)
        self.route_outputs(origin, bus, mul, right, right_belts, column_height)

        poles = [e for e in origin.blueprint.entities[start_entity_index:] if e.name == "small-electric-pole"]
        for a, b in zip(poles, poles[1:]):
            origin.blueprint.add_power_connection(a, b)

        return column_width

    def route_outputs(self, origin, bus, mul, right, right_belts, column_height):
        # Add output(s) to bus
        # TODO: liquids
        for item in mul.outputs:
            rate_out = mul.flow(item).rateOut

            # Find next free bus lane
            if FREE_LANE in bus:
                # Re-use free lane
                lane_index = bus.index(FREE_LANE)
            else:
                # New lane required
                lane_index = len(bus)
                bus.append(FREE_LANE)

            # Route output to bus
            p = origin[right, -column_height + 2]
            route = p.new_route(rate_out)
            route.route(0, column_height - 4).route(-2, 0)

            bus_start = origin[right - 2, lane_y(lane_index)].origin
            if bus_start.y % 2 == 0:
                # Odd lanes can be routed straight in
                route.route_to(bus_start)
            else:
                # Evens need to loop over and in (so the underground can surface first)
                # ---
                #  +/
                #  \/
                route.route_to(bus_start + (0, 1)).route(1, 0).route(0, -1)

            bus[lane_index] = BusLane(item, rate_out, route)


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                highlighter=NullHighlighter(),
                markup=True,
                show_time=False,
                show_level=True,
            )
        ],
    )

    game_mode = "v2.0"
    setGameConfig(game_mode)
    preset = presets.MP_LATE_GAME
    config.machinePrefs.set(preset)
    logger.info("Using game config: %s presets: %s", game_mode, preset)

    # rocketFuel = produce([itm.rocket_fuel @ 6], using=[rcp.advanced_oil_processing]).factory
    # rocketFuel.summary()

    assert rcp

    rate = frac(150, 60)
    # rate = frac(30, 60)
    logger.info("Target SPM: %s/min (%.1f/s)", rate * 60, float(rate))
    solution = produce(
        [
            itm.automation_science_pack @ rate,
            itm.logistic_science_pack @ rate,
            # itm.military_science_pack @ rate,
            # itm.chemical_science_pack @ rate,
            # itm.production_science_pack @ rate,
            # itm.utility_science_pack @ rate,
            # itm.space_science_pack @ rate,
        ],
        using=[
            # rcp.advanced_oil_processing,
            # rcp.coal_liquefaction,
            # rcp.solid_fuel_from_petroleum_gas,
            itm.petroleum_gas,
            itm.lubricant,
            itm.iron_plate,
            itm.copper_plate,
            itm.coal,
            itm.stone,
            itm.water,
            # itm.sulfuric_acid,
            # rcp.solid
        ],
        # using=[rcp.coal_liquefaction],
        roundUp=True,
    )
    logger.info("🏭 [bold]Factory plan[/b]")
    solution.factory.summary()
    for input in solution.factory.inputs:
        flow = solution.factory.flow(input)
        yellows = div(flow.rateIn, 15)
        logger.info("Input: %s %.2f/s (%.1f yellow belts)", input, flow.rateIn, yellows)

    blueprint = Blueprint()
    blueprint.label = "Red and Green"
    blueprint.version = (2, 0)

    LayoutEngine().run(solution.factory, blueprint)

    bstr = blueprint.to_string()
    pyperclip.copy(bstr)
    logger.info("📋 [blue]Blueprint[/] copied to clipboard (%d bytes)", len(bstr))


if __name__ == "__main__":
    main()

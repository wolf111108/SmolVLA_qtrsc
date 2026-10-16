#include <systemc>
#include <iostream>

struct Counter : sc_core::sc_module {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_out<unsigned> count{"count"};
    unsigned value = 0;

    SC_CTOR(Counter) {
        SC_METHOD(tick);
        sensitive << clk.pos();
        dont_initialize();
    }

    void tick() { count.write(++value); }
};          

struct Monitor : sc_core::sc_module {
    sc_core::sc_in<unsigned> count{"count"};
    unsigned samples = 0;
    bool passed = true;

    SC_CTOR(Monitor) { SC_THREAD(check); }

    void check() {
        // Sample halfway between rising edges, after signal delta updates.
        wait(5, sc_core::SC_NS);
        for (unsigned expected = 1; expected <= 10; ++expected) {
            const unsigned actual = count.read();
            std::cout << sc_core::sc_time_stamp() << " count=" << actual << '\n';
            if (actual != expected) {
                passed = false;
                std::cerr << "FAIL: expected " << expected << ", got " << actual << '\n';
            }
            ++samples;
            wait(10, sc_core::SC_NS);
        }
    }
};

int sc_main(int, char**) {
    sc_core::sc_clock clk{"clk", sc_core::sc_time(10, sc_core::SC_NS)};
    sc_core::sc_signal<unsigned> count{"count"};
    Counter counter{"counter"};
    Monitor monitor{"monitor"};
    counter.clk(clk);
    counter.count(count);
    monitor.count(count);

    auto* trace = sc_core::sc_create_vcd_trace_file("counter");
    if (!trace) return 1;
    trace->set_time_unit(1, sc_core::SC_NS);
    sc_core::sc_trace(trace, clk, "clk");
    sc_core::sc_trace(trace, count, "count");
    sc_core::sc_start(100, sc_core::SC_NS);
    sc_core::sc_close_vcd_trace_file(trace);

    if (!monitor.passed || monitor.samples != 10 || count.read() != 10) return 1;
    std::cout << "PASS: counter samples=10 final_count=10 time="
              << sc_core::sc_time_stamp() << '\n';
    return 0;
}

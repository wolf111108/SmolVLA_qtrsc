#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>
#include <algorithm>
#include <array>
#include <iostream>

struct Memory : sc_core::sc_module {
    tlm_utils::simple_target_socket<Memory> socket{"socket"};
    std::array<unsigned char, 64> bytes{};
    const sc_core::sc_time latency{10, sc_core::SC_NS};

    SC_CTOR(Memory) { socket.register_b_transport(this, &Memory::transport); }

    void transport(tlm::tlm_generic_payload& tx, sc_core::sc_time& delay) {
        tx.set_dmi_allowed(false);
        const auto address = tx.get_address();
        const auto length = tx.get_data_length();
        auto* data = tx.get_data_ptr();
        if (!tx.is_read() && !tx.is_write()) {
            tx.set_response_status(tlm::TLM_COMMAND_ERROR_RESPONSE);
        } else if (!data || length == 0) {
            tx.set_response_status(tlm::TLM_GENERIC_ERROR_RESPONSE);
        } else if (tx.get_byte_enable_ptr()) {
            tx.set_response_status(tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE);
        } else if (tx.get_streaming_width() < length) {
            tx.set_response_status(tlm::TLM_BURST_ERROR_RESPONSE);
        } else if (address >= bytes.size() || length > bytes.size() - address) {
            // Subtraction avoids address + length overflowing uint64_t.
            tx.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
        } else {
            auto* destination = bytes.data() + address;
            if (tx.is_write()) std::copy_n(data, length, destination);
            else std::copy_n(destination, length, data);
            delay += latency;
            tx.set_response_status(tlm::TLM_OK_RESPONSE);
        }
    }
};

struct Initiator : sc_core::sc_module {
    tlm_utils::simple_initiator_socket<Initiator> socket{"socket"};
    bool passed = true;
    bool finished = false;
    unsigned transactions = 0;

    SC_CTOR(Initiator) { SC_THREAD(run); }

    void require(bool condition, const char* message) {
        if (!condition) {
            passed = false;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    void transact(tlm::tlm_command command, sc_dt::uint64 address,
                  unsigned char* data, unsigned length,
                  tlm::tlm_response_status expected,
                  unsigned char* byte_enable = nullptr, unsigned width = 4) {
        tlm::tlm_generic_payload tx;
        tx.set_command(command);
        tx.set_address(address);
        tx.set_data_ptr(data);
        tx.set_data_length(length);
        tx.set_streaming_width(width);
        tx.set_byte_enable_ptr(byte_enable);
        tx.set_byte_enable_length(byte_enable ? 1 : 0);
        tx.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
        socket->b_transport(tx, delay);
        require(tx.get_response_status() == expected, "unexpected TLM response");
        require(delay == (expected == tlm::TLM_OK_RESPONSE
                             ? sc_core::sc_time(10, sc_core::SC_NS)
                             : sc_core::SC_ZERO_TIME), "unexpected annotated delay");
        // Target annotates; initiator advances simulation time exactly once.
        if (delay != sc_core::SC_ZERO_TIME) wait(delay);
        ++transactions;
        std::cout << sc_core::sc_time_stamp() << " address=" << address
                  << " response=" << tx.get_response_string() << '\n';
    }

    void run() {
        // Explicit byte order makes the example independent of host endianness.
        std::array<unsigned char, 4> written{0x78, 0x56, 0x34, 0x12};
        std::array<unsigned char, 4> readback{};
        transact(tlm::TLM_WRITE_COMMAND, 16, written.data(), 4, tlm::TLM_OK_RESPONSE);
        transact(tlm::TLM_READ_COMMAND, 16, readback.data(), 4, tlm::TLM_OK_RESPONSE);
        require(readback == written, "readback differs from written bytes");

        std::array<unsigned char, 4> rejected{0xff, 0xff, 0xff, 0xff};
        unsigned char enable = 0xff;
        transact(tlm::TLM_WRITE_COMMAND, 63, rejected.data(), 4,
                 tlm::TLM_ADDRESS_ERROR_RESPONSE);
        transact(tlm::TLM_WRITE_COMMAND, 16, rejected.data(), 4,
                 tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE, &enable);
        transact(tlm::TLM_WRITE_COMMAND, 16, rejected.data(), 4,
                 tlm::TLM_BURST_ERROR_RESPONSE, nullptr, 2);
        transact(tlm::TLM_WRITE_COMMAND, 16, nullptr, 4, tlm::TLM_GENERIC_ERROR_RESPONSE);
        transact(tlm::TLM_IGNORE_COMMAND, 16, rejected.data(), 4,
                 tlm::TLM_COMMAND_ERROR_RESPONSE);
        transact(tlm::TLM_WRITE_COMMAND, 16, rejected.data(), 0,
                 tlm::TLM_GENERIC_ERROR_RESPONSE);
        readback.fill(0);
        transact(tlm::TLM_READ_COMMAND, 16, readback.data(), 4, tlm::TLM_OK_RESPONSE);
        require(readback == written, "rejected writes changed valid memory");
        require(sc_core::sc_time_stamp() == sc_core::sc_time(30, sc_core::SC_NS),
                "simulation should consume exactly 30 ns");
        finished = true;
        sc_core::sc_stop();
    }
};

int sc_main(int, char**) {
    Memory memory{"memory"};
    Initiator initiator{"initiator"};
    initiator.socket.bind(memory.socket);
    sc_core::sc_start();
    if (!initiator.passed || !initiator.finished || initiator.transactions != 9) return 1;
    std::cout << "PASS: tlm_memory transactions=9 time="
              << sc_core::sc_time_stamp() << '\n';
    return 0;
}

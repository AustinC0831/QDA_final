/****************************************************************************
  PackageName  [ tensor ]
  Synopsis     [ Define tensor package commands ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#include <cstddef>
#include <string>

#include "./tensor_mgr.hpp"
#include "cli/cli.hpp"
#include "util/data_structure_manager_common_cmd.hpp"
#include "util/phase.hpp"
#include "util/text_format.hpp"
// #include "./tensor_decompose.hpp"

using namespace dvlab::argparse;
using dvlab::CmdExecResult;
using dvlab::Command;

namespace qsyn::tensor
{

    struct two_level_matrix
    {
        QTensor<double> given;
        size_t i, j; // i < j
        two_level_matrix(QTensor<double> U) : given(U) {}
    };

    std::vector<two_level_matrix> decompose(QTensor<double> *t)
    {
        std::vector<two_level_matrix> two_level_chain;
        fmt::println("start decomposing...");

        using namespace std::literals;
        size_t num = 0;

        (*t) = {
            {0.353553 + 0.i, 0. + 0.i, -0.612372 + 0.i, 0.707107 + 0.i},
            {0. + -0.866025i, 0. + 0.i, 0. - 0.5i, 0. + 0.i},
            {0. + 0.i, 0. + 1.i, 0. + 0.i, 0. + 0.i}, // test
            {-0.353553 + 0.i, 0. + 0.i, 0.612372 + 0.i, 0.707107 + 0.i},
        };
        // fmt::println("今天拆解的矩陣是:");
        // fmt::println("{}", *t);

        size_t s = (*t).shape()[0];
        fmt::println("shape : {} * {}", s, s);

        QTensor<double>
            U = QTensor<double>::identity((int)round(std::log2(s)));
        // QTensor<double> I = QTensor<double>::identity(1);
        // QTensor<double> U = tensor_product_pow(I, (int)round(std::log2(s)));
        U.reshape({s, s});

        for (size_t i = 0; i < s; i++)
        {
            for (size_t j = i + 1; j < s; j++)
            {
                // if (std::abs((*t)(i, i).real() - 1) < 1e-6 && std::abs((*t)(i, i).imag()) < 1e-6) {  // maybe use e-6 approx.
                //     if (std::abs((*t)(j, i).real()) < 1e-6 && std::abs((*t)(j, i).imag()) < 1e-6) {
                //         fmt::println("skip cuz (1,0) {},{}", i, j);
                //         continue;
                //     }
                // }
                // if (std::abs((*t)(i, i).real()) < 1e-6 && std::abs((*t)(i, i).imag()) < 1e-6) {  // maybe use e-6 approx.
                //     if (std::abs((*t)(j, i).real()) < 1e-6 && std::abs((*t)(j, i).imag()) < 1e-6) {
                //         fmt::println("skip cuz (0,0) {},{}", i, j);
                //         continue;
                //     }
                // }

                if (std::abs((*t)(j, i).real()) < 1e-6 && std::abs((*t)(j, i).imag()) < 1e-6)
                {
                    fmt::println("skip cuz U({},{}) = 0", j, i);
                    continue;
                }

                fmt::println("拆! i = {}, j = {}", i, j);
                num++;

                double u = std::sqrt(std::norm((*t)(i, i)) + std::norm((*t)(j, i)));
                // fmt::println("u = {}", u);

                using namespace std::literals;

                for (size_t x = 0; x < s; x++)
                {
                    for (size_t y = 0; y < s; y++)
                    {
                        if (x == y)
                        {
                            if (x == i)
                            {
                                U(x, y) = (std::conj((*t)(i, i))) / u;
                            }
                            else if (x == j)
                            {
                                U(x, y) = (*t)(i, i) / u;
                            }
                            else
                            {
                                U(x, y) = 1.0 + 0.i;
                            }
                        }
                        else if (x == j && y == i)
                        {
                            U(x, y) = (-1. + 0.i) * (*t)(j, i) / u;
                        }
                        else if (x == i && y == j)
                        {
                            U(x, y) = (std::conj((*t)(j, i))) / u;
                        }
                        else
                        {
                            U(x, y) = 0. + 0.i;
                        }
                    }
                }
                // fmt::println("U{}", num);
                // fmt::println("{}", U);

                // take a, b from (*t)(i,i) & (*t)(j,i)
                // std::complex<double> a = (*t)(i, i);
                // std::complex<double> b = (*t)(j, i);

                (*t) = tensordot(U, *t, {1}, {0});
                // fmt::println("目前結果:");
                // fmt::println("{}", *t);

                // U(ii/ij/ji/jj) adjust
                QTensor<double>
                    CU = QTensor<double>::identity(1);

                CU(0, 0) = std::conj(U(i, i));
                CU(0, 1) = std::conj(U(j, i));
                CU(1, 0) = std::conj(U(i, j));
                CU(1, 1) = std::conj(U(j, j));

                two_level_matrix m(CU);
                m.i = i;
                m.j = j;
                two_level_chain.push_back(m);

                // fmt::println("CU{}", num);
                // fmt::println("{}", two_level_chain[num - 1].given);

                // check if *t is 2-level ,end

                size_t count = 0;
                size_t c_i = 1073741824; // 2^30
                size_t c_j = 1073741824; // 2^30
                bool is_two_level = true;

                for (size_t x = 0; x < s; x++)
                { // check all

                    if (!is_two_level)
                        break;

                    for (size_t y = 1; y < x; y++)
                    {
                        if (std::abs((*t)(y, x).real()) > 1e-6 || std::abs((*t)(y, x).imag()) > 1e-6)
                        { // entry(j,i)!=0
                            count++;

                            if (count == 1)
                            {
                                c_i = y;
                                c_j = x;
                            }
                            else
                            {
                                is_two_level = false;
                            }
                        }
                    }
                }

                for (size_t x = 0; x < s; x++)
                {
                    if ((c_i == 1073741824) || !is_two_level)
                        break;
                    if (std::abs((*t)(x, x).real() - 1) > 1e-6 || std::abs((*t)(x, x).imag()) > 1e-6)
                    { // entry(x,x)!=1

                        if (x != c_i && x != c_j)
                        {
                            is_two_level = false;
                        }
                    }
                }
                if (is_two_level == true)
                {
                    CU(0, 0) = std::conj((*t)(c_i, c_i));
                    CU(0, 1) = std::conj((*t)(c_j, c_i));
                    CU(1, 0) = std::conj((*t)(c_i, c_j));
                    CU(1, 1) = std::conj((*t)(c_j, c_j));
                    two_level_matrix m(CU);
                    m.i = c_i;
                    m.j = c_j;
                    two_level_chain.push_back(m);
                    fmt::println("find *t has been 2 level matrix, put it in the chain");
                    // fmt::println("{}", CU);
                    // fmt::println("in the chain");

                    return two_level_chain;
                }
            }
        }
        return two_level_chain;
    }

    ArgType<size_t>::ConstraintType valid_tensor_id(TensorMgr const &tensor_mgr)
    {
        return [&](size_t const &id)
        {
            if (tensor_mgr.is_id(id))
                return true;
            spdlog::error("Cannot find tensor with ID {}!!", id);
            return false;
        };
    }

    Command tensor_print_cmd(TensorMgr &tensor_mgr)
    {
        return {"print",
                [&](ArgumentParser &parser)
                {
                    parser.description("print info of Tensor");

                    parser.add_argument<size_t>("id")
                        .constraint(valid_tensor_id(tensor_mgr))
                        .nargs(NArgsOption::optional)
                        .help("if specified, print the tensor with the ID");
                },
                [&](ArgumentParser const &parser)
                {
                    if (parser.parsed("id"))
                    {
                        fmt::println("{}", *tensor_mgr.find_by_id(parser.get<size_t>("id")));
                    }
                    else
                    {
                        fmt::println("{}", *tensor_mgr.get());
                    }
                    return CmdExecResult::done;
                }};
    }

    Command tensor_decompose_cmd(TensorMgr &tensor_mgr)
    {
        return {"decompose",
                [&](ArgumentParser &parser)
                {
                    parser.description("Decompose the unitary matrix into multi two level matrix");

                    parser.add_argument<size_t>("id")
                        .constraint(valid_tensor_id(tensor_mgr))
                        .nargs(NArgsOption::optional)
                        .help("if specified, decompose the tensor with the ID");
                },
                [&](ArgumentParser const &parser)
                {
                    QTensor<double> *tensor;

                    if (parser.parsed("id"))
                    {
                        tensor = tensor_mgr.find_by_id(parser.get<size_t>("id"));
                    }
                    else
                    {
                        tensor = tensor_mgr.get();
                    }

                    std::vector<two_level_matrix> tlc = decompose(tensor);

                    for (size_t i = 0; i < tlc.size(); i++)
                    {
                        fmt::println("U{}' = {}", i, tlc[i].given);
                        fmt::println("work on i = {}, j = {}", tlc[i].i, tlc[i].j);
                    }

                    fmt::println("final U(*t) : {}", *tensor);
                    return CmdExecResult::done;
                }};
    }

    Command tensor_adjoint_cmd(TensorMgr &tensor_mgr)
    {
        return {"adjoint",
                [&](ArgumentParser &parser)
                {
                    parser.description("adjoint the specified tensor");

                    parser.add_argument<size_t>("id")
                        .constraint(valid_tensor_id(tensor_mgr))
                        .nargs(NArgsOption::optional)
                        .help("the ID of the tensor");
                },
                [&](ArgumentParser const &parser)
                {
                    if (parser.parsed("id"))
                    {
                        tensor_mgr.find_by_id(parser.get<size_t>("id"))->adjoint();
                    }
                    else
                    {
                        tensor_mgr.get()->adjoint();
                    }
                    return CmdExecResult::done;
                }};
    }
    Command tensor_equivalence_check_cmd(TensorMgr &tensor_mgr)
    {
        return {"equiv",
                [&](ArgumentParser &parser)
                {
                    parser.description("check the equivalency of two stored tensors");

                    parser.add_argument<size_t>("ids")
                        .nargs(1, 2)
                        .constraint(valid_tensor_id(tensor_mgr))
                        .help("Compare the two tensors. If only one is specified, compare with the tensor on focus");
                    parser.add_argument<double>("-e", "--epsilon")
                        .metavar("eps")
                        .default_value(1e-6)
                        .help("output \"equivalent\" if the Frobenius inner product is at least than 1 - eps (default: 1e-6)");
                    parser.add_argument<bool>("-s", "--strict")
                        .help("requires global scaling factor to be 1")
                        .action(store_true);
                },
                [&](ArgumentParser const &parser)
                {
                    auto ids = parser.get<std::vector<size_t>>("ids");
                    auto eps = parser.get<double>("--epsilon");
                    auto strict = parser.get<bool>("--strict");

                    QTensor<double> *tensor1 = nullptr;
                    QTensor<double> *tensor2 = nullptr;
                    if (ids.size() == 2)
                    {
                        tensor1 = tensor_mgr.find_by_id(ids[0]);
                        tensor2 = tensor_mgr.find_by_id(ids[1]);
                    }
                    else
                    {
                        tensor1 = tensor_mgr.get();
                        tensor2 = tensor_mgr.find_by_id(ids[0]);
                    }

                    bool equiv = is_equivalent(*tensor1, *tensor2, eps);
                    auto const norm = global_norm(*tensor1, *tensor2);
                    auto const phase = global_phase(*tensor1, *tensor2);

                    if (strict)
                    {
                        if (norm > 1 + eps || norm < 1 - eps || phase != dvlab::Phase(0))
                        {
                            equiv = false;
                        }
                    }
                    using namespace dvlab;
                    if (equiv)
                    {
                        fmt::println("{}", fmt_ext::styled_if_ansi_supported("Equivalent", fmt::fg(fmt::terminal_color::green) | fmt::emphasis::bold));
                        fmt::println("- Global Norm : {:.6}", norm);
                        fmt::println("- Global Phase: {}", phase);
                    }
                    else
                    {
                        fmt::println("{}", fmt_ext::styled_if_ansi_supported("Not Equivalent", fmt::fg(fmt::terminal_color::red) | fmt::emphasis::bold));
                    }

                    return CmdExecResult::done;
                }};
    }

    Command tensor_cmd(TensorMgr &tensor_mgr)
    {
        using namespace dvlab::utils;
        auto cmd = mgr_root_cmd(tensor_mgr);
        cmd.add_subcommand(mgr_list_cmd(tensor_mgr));
        cmd.add_subcommand(tensor_print_cmd(tensor_mgr));
        cmd.add_subcommand(mgr_checkout_cmd(tensor_mgr));
        cmd.add_subcommand(mgr_delete_cmd(tensor_mgr));
        cmd.add_subcommand(tensor_adjoint_cmd(tensor_mgr));
        cmd.add_subcommand(tensor_equivalence_check_cmd(tensor_mgr));
        cmd.add_subcommand(tensor_decompose_cmd(tensor_mgr));

        return cmd;
    }

    bool add_tensor_cmds(dvlab::CommandLineInterface &cli, TensorMgr &tensor_mgr)
    {
        if (!cli.add_command(tensor_cmd(tensor_mgr)))
        {
            spdlog::error("Registering \"tensor\" commands fails... exiting");
            return false;
        }
        return true;
    }

} // namespace qsyn::tensor

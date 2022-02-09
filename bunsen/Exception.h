#ifndef BUNSEN_EXCEPTION_H
#define BUNSEN_EXCEPTION_H

#include <string>

#define BUNSEN_MAKE_EXCEPTION(name, defaultStr)             \
    class name : public std::exception {                    \
    private:                                                \
        std::string m_What;                                 \
    public:                                                 \
        explicit name(std::string what = defaultStr)        \
        : m_What(std::move(what)) {}                        \
                                                            \
        inline const char *what() const noexcept {          \
            return this->m_What.c_str();                    \
        }                                                   \
    }

#endif //BUNSEN_EXCEPTION_H
